# Parameter Service
About:

This service provides parameter subscription function. It can be accessed from inside or outside frl cluster.

It uses multicast inside the cluster and TCP outside the cluster to publish subscribed parameter.

## APIs
The service now exposes four interfaces which accept requests in json format:

* connect

Establish connect between the service and client.

Return an 'ok' status and a unique 'client_id'.

    request = {
        'type': 'connect'
    }

    response = {
        'status': 'ok',
        'client_id': ,
    }


* subscribe

Subscribe parameter defined by experiment_name, trial_name, policy_name, tag and user_name.

Return an 'ok' status, the address which is used to publish the parameter and the key of subscription if succeeded.

Return an 'error' status, and the comment if failed.

    request = {
        'type': 'subscribe',
        'client_id': ,
        'client_type': ,
        'user_name': ,
        'experiment_name': ,
        'trial_name': ,
        'policy_name': ,
        'tag_name': ,
    }

    success_response = {
        'status': 'ok',
        'pub_address': ,
        'sub_key': ,
    }

    error_response = {
        'status': 'error',
        'comment':
    }

* unsubscribe

Unsubscribe a parameter which have been subscribed.

Return an 'ok' status if succeeded.

Return an 'error' status, and the comment if failed.

    request = {
        'type': 'unsubscribe',
        'client_id': ,
        'sub_key': ,
    }

    success_response = {
        'status': 'ok',
    }

    error_response = {
        'status': 'error',
        'comment':
    }

* touch

Let the service know the client is alive.

The service would clean all the client's subscriptions if touch request isn't received in 120 seconds.

Return an 'ok' status if succeeded.

Return an 'error' status, and the comment if failed.

    request = {
        'type': 'touch',
        'client_id': ,
    }

    success_response = {
        'status': 'ok',
    }

    error_response = {
        'status': 'error',
        'comment': '',
    }

## Use Example
Provided client can be used to send request to parameter service.
```commandline
from distributed.service.parameter_service_client import make_client
client = make_client()
client.connect("10.210.13.107:3001")
# the 'user_name' is optional, use getpass.getuser() by default
sub_key = client.subscribe('experiment_name', 'trial_name', 'policy_name', 'tag', callback_fn, 'user_name')
# do somthing other
client.ubsubscribe(sub_key)
```
