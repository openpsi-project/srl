from typing import Any, Tuple, Optional, List, Callable, Dict, Union
import dataclasses
import logging
import os
import time
import torch
import shutil
import socket

import api.config
import base.names
import base.timeutil

GRACE_PERIOD = 3  # seconds
PARAMETER_CLIENT_RECV_TIMEO = 5000  # in ms
PARAMETER_SUBSCRIPTION_ID_HEXLENGTH = 16

logger = logging.getLogger("param-db")


class ParameterDBClient:
    """Defines the communication between an user and the parameter database (aka parameter server).

    Concepts:
    + policy, or policy type: a code implementing //algorithm/policy.py.
    + policy name: the trial-level unique name to identify policy parameters. The policy workers and the
      trainer workers save and load trained policies (essentially the parameters) using this name.
    + policy version: each policy, identified by the name, can have different versions as the training time
      goes on. The version can be used to load a specific frozen policy.
    + tag: a customized human-readable alias for a policy version.
    + identifier: either the raw policy version id, or a tag.
    """

    def __init__(self, experiment_name, trial_name):
        """Creates a client on a trial namespace.
        """
        self.experiment_name = experiment_name
        self.trial_name = trial_name
        self.namespace = self.experiment_name + "/" + self.trial_name

    def clone(self, source_experiment_name, source_trial_name):
        """Clones all policies and their tags from another trial namespace.
        """
        raise NotImplementedError()

    def list_names(self):
        """List all policy_names with available tags.
        """
        raise NotImplementedError()

    def clear(self, name=None):
        """Removes all saved versions of a policy. If name is None, removes all policies under the current
        trial namespace.
        """
        raise NotImplementedError()

    def gc(self, name, max_untagged_version_count=None, max_untagged_version_ttl=None):
        """Removes policies that are outdated.
        """
        raise NotImplementedError()

    def push(self,
             name,
             checkpoint,
             version: str,
             tags: Union[None, str, List[str]] = None,
             metadata: Dict[str, Any] = None) -> str:
        """Pushes a set of new parameters to the database.

        This is typically called by the trainer to publish the newest model parameters. An optional tag can
        be added to the version; also the "latest" tag will always be added.

        Args:
            name (str): Name of the policy to push.
            checkpoint (Any): The parameters of the policy version.
            version (str): An indicator that tells the version of the checkpoint. For example it can be the number
              of gradient steps performed to get this checkpoint.
            tags (str or List[str]): An optional tag or a list of tags to associate with the policy.
            metadata (Dict[str, Any]): An optional collection of metadata associated with the policy. Those
              are useful for evaluating and selecting policy versions on demand. Examples: training steps,
              evaluation win rate, other training/testing properties, etc.
        Returns:
            version (str): The auto-generated policy version that can be used to identify the policy.
        """
        raise NotImplementedError()

    def tag(self, name, identifier, new_tag):
        """Associates a policy version with a new tag. If the tag already exists, it will be replaced with the
        provided version.

        Args:
            name (str): Name of the policy.
            identifier (str): The policy version or a tag.
            new_tag (str): The new tag to associate with the policy version.
        """
        raise NotImplementedError()

    def get(self, name, identifier="latest", block: bool = False, retry_times=60, mode="pytorch") -> Any:
        """Acquires a set of parameters from the database.

        This is typically called by the policy worker to update its model parameters with the
        newest.

        Args:
            name (str): Name of the policy to get.
            identifier (str): The version or tag to identify a policy with the specified name.
            block (bool): Whether the get operation is blocking.
            retry_times (int): times to retry in case of expected exceptions.
            mode: pytorch or bytes
        Returns:
            The restored parameters of the specified version of the policy.
        Raises:
            FileNotFoundError if parameter of `identifier` is not found.
        """
        raise NotImplementedError()

    def list_versions(self, name) -> List[str]:
        """Lists policy versions, sorted by time.

        Returns:
            List of versions.
        """
        raise NotImplementedError()

    def list_tags(self, name) -> List[Tuple[str, str]]:
        """Lists tagged policy versions.

        Returns:
            Pairs of (tag, version).
        """
        raise NotImplementedError()

    def has_tag(self, name, tag) -> bool:
        """Check whether a tag exists for a policy.

        Returns:
            Check result (bool).
        """
        raise NotImplementedError()

    def version_of(self, name, identifier) -> int:
        """Get the version of an identifier.
        Args:
            name: name of the policy
            identifier: identifier of the policy version.
        """
        raise NotImplementedError()

    def update_metadata(self, name: str, version: str, metadata: dict, **kwargs):
        """Update the metadata of a saved parameter.
        Args:
            name: policy_name
            version: version of the parameter
            metadata: new metadata.
        No Returns.
        """
        raise NotImplementedError()


class PytorchFilesystemParameterDB(ParameterDBClient):
    if socket.gethostname().startswith("frl") or socket.gethostname().startswith("dev"):
        ROOT = f"/data/marl/checkpoints/{base.names.USER_NAMESPACE}"
    else:
        ROOT = "/tmp/marl_checkpoints"

    @dataclasses.dataclass
    class Subscription:
        topic: str
        callback: Callable

    @staticmethod
    def purge(experiment_name, trial_name):
        ckpt_dir = os.path.join(PytorchFilesystemParameterDB.ROOT, experiment_name, trial_name)
        if os.path.exists(ckpt_dir):
            shutil.rmtree(ckpt_dir)

    def __init__(self, experiment_name, trial_name):
        super().__init__(experiment_name, trial_name)
        self.__workdir = os.path.join(self.ROOT, experiment_name, trial_name)
        self.__last_time = None
        self.__last_index = None
        os.makedirs(self.__workdir, exist_ok=True, mode=0o775)
        self.__sub_thread = None
        self.__stop_sub = False

    def list_names(self):
        return [n for n in os.listdir(self.__workdir) if len(self.list_tags(n)) > 0]

    def __path_of(self, name, identifier):
        return os.path.join(self.__workdir, name, identifier)

    def __is_tag(self, name, tag):
        return os.path.islink(self.__path_of(name, tag))

    def __list_all(self, name):
        return [f for f in os.listdir(os.path.join(self.__workdir, name)) if not f.endswith(".tmp")]

    def _has_version(self, name, version):
        return os.path.exists(self.__path_of(name, version)) and not self.__is_tag(name, version)

    def _has_tag(self, name, tag):
        return os.path.exists(self.__path_of(name, tag)) and self.__is_tag(name, tag)

    def clone(self, source_experiment_name, source_trial_name):
        raise NotImplementedError()

    def clear(self, name=None):
        if name is None:
            shutil.rmtree(self.__workdir)
        else:
            shutil.rmtree(os.path.join(self.__workdir, name))

    def _list_outdated_version(self, name, keep_count, keep_ttl):
        if keep_ttl is not None:
            raise NotImplementedError()
        has_tag = set(v for _, v in self.list_tags(name))
        untagged = [v for v in self.list_versions(name) if v not in has_tag]
        if keep_count is not None:
            return untagged[:-keep_count]
        return []

    def _remove(self, name, version):
        os.remove(self.__path_of(name, version))

    def gc(self, name, max_untagged_version_count=None, max_untagged_version_ttl=None):
        gv = self._list_outdated_version(name, max_untagged_version_count, max_untagged_version_ttl)
        count = 0
        for v in gv:
            self._remove(name, v)
            count += 1
        logger.debug("Removed %d outdated versions for policy %s", count, name)

    def push(self, name, checkpoint, version: str, tags=None, metadata=None):
        assert metadata is None, "Use param_db of type `METADATA` to support metadata query."
        path = self.__path_of(name, version)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(checkpoint, path)
        self.tag(name, version, "latest")
        if tags is not None:
            if isinstance(tags, str):
                tags = [tags]
            for tag in tags:
                self.tag(name, version, tag)
        return version

    def __readlink(self, link_path):
        # TODO: upgrade mu01 RedHat.
        for i in range(10):
            try:
                return os.readlink(link_path)
            except FileNotFoundError as e:
                raise e
            except OSError:
                time.sleep(0.005)
        else:
            raise OSError("Read symlink Failed.")

    def tag(self, name, identifier, new_tag):
        if self.__is_tag(name, identifier):
            identifier = self.__readlink(self.__path_of(name, identifier))
        tmp_path = self.__path_of(name, new_tag + ".tmp")
        new_path = self.__path_of(name, new_tag)
        os.symlink(identifier, tmp_path)
        shutil.move(src=tmp_path, dst=new_path)

    def has_tag(self, name, identifier):
        path = self.__path_of(name, identifier)
        return os.path.lexists(path)

    def version_of(self, name, identifier) -> int:
        # TODO: add version to param_db API.
        ckpt = self.get(name, identifier)
        return ckpt.get("steps", -1)

    def get(self, name, identifier="latest", block=False, retry_times=60, mode="pytorch"):
        # TODO: refactor recursive call.
        path = self.__path_of(name, identifier)
        if retry_times < 0:
            raise FileNotFoundError(f"Read checkpoint failed {name} {identifier}.")
        if block and not os.path.lexists(path):
            time.sleep(1)
            return self.get(name, identifier=identifier, block=block, retry_times=retry_times - 1, mode=mode)
        try:
            # TODO: upgrade mu01 RedHat.
            if mode == "pytorch":
                return torch.load(path, map_location="cpu")
            elif mode == "bytes":
                with open(path, "rb") as f:
                    return f.read()
            else:
                raise NotImplementedError()
        except FileNotFoundError as e:
            raise e
        except OSError as e:
            time.sleep(0.005)
            logger.error(f"Read parameter failed due to {e}. Retrying...")
            return self.get(name, identifier, retry_times=retry_times - 1, mode=mode)

    @staticmethod
    def get_file(file_path):
        return torch.load(file_path, map_location="cpu")

    def list_versions(self, name):
        rs = []
        if not os.path.isdir(os.path.join(self.__workdir, name)):
            return rs
        for version in self.__list_all(os.path.join(self.__workdir, name)):
            if not self.__is_tag(name, version):
                rs.append(version)
        rs.sort(key=lambda x: int(x))  # version is number of steps.
        return rs

    def list_tags(self, name):
        rs = []
        if not os.path.isdir(os.path.join(self.__workdir, name)):
            return rs
        for tag in self.__list_all(os.path.join(self.__workdir, name)):
            if self.__is_tag(name, tag):
                rs.append((tag, self.__readlink(self.__path_of(name, tag))))
        return rs

    def update_metadata(self, name: str, version: str, metadata: dict, **kwargs):
        raise NotImplementedError()


class LocalTestPytorchParamDB(ParameterDBClient):
    """Testing use only.
    """
    import tempfile
    ckpt_path = tempfile.NamedTemporaryFile().name

    def clone(self, source_experiment_name, source_trial_name):
        raise NotImplementedError()

    def list_names(self):
        raise NotImplementedError()

    def clear(self, name=None):
        raise NotImplementedError()

    def gc(self, name, max_untagged_version_count=None, max_untagged_version_ttl=None):
        raise NotImplementedError()

    def push(self,
             name,
             checkpoint,
             version: str,
             tags: Union[None, str, List[str]] = None,
             metadata: Dict[str, Any] = None) -> str:
        torch.save(checkpoint, LocalTestPytorchParamDB.ckpt_path)
        return ""

    def tag(self, name, identifier, new_tag):
        raise NotImplementedError()

    def get(self, name, identifier="latest", block: bool = False, retry_times=60, mode="pytorch") -> Any:
        if mode == "pytorch":
            return torch.load(LocalTestPytorchParamDB.ckpt_path, map_location="cpu")
        elif mode == "bytes":
            with open(LocalTestPytorchParamDB.ckpt_path, "rb") as f:
                return f.read()
        else:
            raise NotImplementedError()

    def list_tags(self, name) -> List[Tuple[str, str]]:
        raise NotImplementedError()

    def has_tag(self, name, tag) -> bool:
        raise NotImplementedError()

    def version_of(self, name, identifier) -> int:
        raise NotImplementedError()

    def subscribe(self, name, callback_fn, tag="latest"):
        raise NotImplementedError()

    def poll(self):
        raise NotImplementedError()

    def update_metadata(self, name: str, version: str, metadata: dict, **kwargs):
        raise NotImplementedError()

    def list_versions(self, name) -> List[str]:
        raise NotImplementedError()


def make_db(spec: api.config.ParameterDB, worker_info: Optional[api.config.WorkerInformation] = None):
    if spec.type_ == api.config.ParameterDB.Type.FILESYSTEM:
        return PytorchFilesystemParameterDB(experiment_name=worker_info.experiment_name,
                                            trial_name=worker_info.trial_name)
    elif spec.type_ == api.config.ParameterDB.Type.LOCAL_TESTING:
        return LocalTestPytorchParamDB("basic", "testing")
    else:
        raise NotImplementedError("Parameter db {} not implemented".format(spec.type_))
