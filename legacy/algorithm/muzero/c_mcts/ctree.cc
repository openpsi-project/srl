#include "ctree.h"
#include <assert.h>
#include <iostream>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace tools {

MinMaxStats::MinMaxStats() {
  this->maximum = FLOAT_MIN;
  this->minimum = FLOAT_MAX;
  this->value_delta_max = 0.;
}

MinMaxStats::~MinMaxStats() {}

void MinMaxStats::set_delta(float value_delta_max) {
  this->value_delta_max = value_delta_max;
}

void MinMaxStats::update(float value) {
  if (value > this->maximum) {
    this->maximum = value;
  }
  if (value < this->minimum) {
    this->minimum = value;
  }
}

void MinMaxStats::clear() {
  this->maximum = FLOAT_MIN;
  this->minimum = FLOAT_MAX;
}

float MinMaxStats::normalize(float value) {
  float norm_value = value;
  float delta = this->maximum - this->minimum;
  if (delta > 0) {
    if (delta < this->value_delta_max) {
      norm_value = (norm_value - this->minimum) / this->value_delta_max;
    } else {
      norm_value = (norm_value - this->minimum) / delta;
    }
  }
  return norm_value;
}

} // namespace tools

namespace mcts {
Node::Node(int id, Node *parent, float value_prefix,
           const std::vector<float> &policy) {
  this->action_num = policy.size();
  this->visit_count = 0;
  this->is_reset = 0;
  this->value_sum = 0;
  this->value_prefix = value_prefix;

  this->id = id;
  this->parent = parent;
  if (this->parent != nullptr)
    this->depth = this->parent->depth + 1;
  else
    this->depth = 0; // is root

  this->children.clear();
  this->explored_actions.clear();
  this->prior.clear();
  for (int a = 0; a < this->action_num; a++) {
    this->children.push_back(nullptr);
    this->prior.push_back(policy[a]);
  }
}
Node::~Node() {}

float Node::get_mean_q(int isRoot, float parent_q, float discount) {
  // seems such mean q is not consistent with muzero paper?
  // TODO: check EfficientZero paper
  float total_unsigned_q = 0.0;
  int total_visits = 0;
  for (auto a : this->explored_actions) {
    Node *child = this->get_child(a);
    assert(child != nullptr and child->visit_count > 0);
    float true_reward =
        child->value_prefix - this->value_prefix * (1 - this->is_reset);
    float qsa = true_reward + discount * child->value();
    total_unsigned_q += qsa;
    total_visits++;
  }

  float mean_q = 0.0;
  if (isRoot && total_visits > 0)
    mean_q = total_unsigned_q / total_visits;
  else
    mean_q = (parent_q + total_unsigned_q) / (total_visits + 1);
  return mean_q;
}
float Node::value() {
  if (this->visit_count == 0)
    return 0.0;
  else
    return this->value_sum / this->visit_count;
}
Node *Node::get_child(int action) { return this->children[action]; }
std::vector<int> Node::get_children_distribution() {
  std::vector<int> distribution;
  for (int a = 0; a < this->action_num; a++) {
    Node *child = this->get_child(a);
    if (child != nullptr)
      distribution.push_back(child->visit_count);
    else
      distribution.push_back(0);
  }
  return distribution;
}

int Node::select_action(tools::MinMaxStats &min_max_stats, int pb_c_base,
                        float pb_c_init, float discount, float mean_q) {
  float max_score = FLOAT_MIN;
  const float epsilon = 0.000001;
  std::vector<int> max_index_lst;
  std::vector<float> score;
  score.clear();
  max_index_lst.clear();

  // std::cout<<"Node "<<this->id<<" select action:"<<std::endl;

  for (int a = 0; a < this->action_num; a++) {
    if (this->available_action[a] == 0) {
      // this node is root and the action is unavailable
      score.push_back(FLOAT_MIN * 2);
      continue;
    }
    Node *child = this->get_child(a);

    // compute ucb score
    float pb_c = 0.0, prior_score = 0.0, value_score = 0.0;
    pb_c = log((this->visit_count - 1 + pb_c_base + 1) / pb_c_base) + pb_c_init;
    pb_c *= sqrt(this->visit_count - 1) /
            (child == nullptr ? 1 : child->visit_count + 1);

    prior_score = pb_c * this->prior[a];
    if ((child == nullptr) || (child->visit_count == 0))
      value_score = mean_q;
    else {
      float true_reward =
          child->value_prefix - this->value_prefix * (1 - this->is_reset);
      value_score = true_reward + discount * child->value();
    }

    value_score = min_max_stats.normalize(value_score);

    if (value_score < 0)
      value_score = 0;
    if (value_score > 1)
      value_score = 1;

    float ucb_value = prior_score + value_score;

    // std::cout<<prior_score<<" "<<value_score<<std::endl;

    score.push_back(ucb_value);
    if (ucb_value > max_score)
      max_score = ucb_value;
  }

  for (int a = 0; a < this->action_num; a++) {
    if (score[a] >= max_score - epsilon)
      max_index_lst.push_back(a);
  }

  int action = 0;
  if (max_index_lst.size() > 0)
    action = max_index_lst[rand() % max_index_lst.size()];
  // std::cout<<"max_index_lst";
  // for(auto x: max_index_lst)
  // std::cout<<" "<<x;
  // std::cout<<std::endl;
  // std::cout<<"score";
  // for(int a = 0; a < this->action_num; a++)
  // std::cout<<" "<<score[a];
  // std::cout<<std::endl;
  return action;
}
/*
    // MuZero related
    float pb_c_base, pb_c_init, discount;
    tools::MinMaxStats min_max_stats;

    // tree structure
    Node* root;
    std::vector<Node*> node_list;
    int num_nodes;

    // MCTS
    Node* query_node;
    int query_action;*/

Tree::Tree() {}
Tree::~Tree() { this->clear(); }

void Tree::clear() {
  this->min_max_stats.clear();
  this->num_nodes = 0;
  this->root = nullptr;
  if (this->node_list.size() > 0) {
    int n = this->node_list.size();
    for (int i = 0; i < n; i++)
      delete this->node_list[i];
  }
  this->node_list.clear();
  this->query_node = nullptr;
  this->query_action = -1;
}

void Tree::reset(int pb_c_base, float pb_c_init, float discount,
                 float value_delta_max, float value_prefix,
                 const std::vector<float> &policy,
                 const std::vector<int> &available_action) {
  this->clear();
  this->pb_c_base = pb_c_base;
  this->pb_c_init = pb_c_init;
  this->discount = discount;

  this->root = new Node(0, nullptr, value_prefix, policy);
  this->num_nodes++;
  this->node_list.push_back(this->root);
  this->root->visit_count++;

  this->min_max_stats.set_delta(value_delta_max);
  this->root->available_action.assign(available_action.begin(),
                                      available_action.end());
}

void Tree::traverse(std::vector<int> &ret) {
  assert(this->query_node == nullptr && this->query_action == -1);

  Node *last_node = this->root;
  Node *node = this->root;
  float parent_q = 0.0;
  int last_action = 0;
  int is_root = 1;

  // std::vector<std::pair<Node *, int>> path;
  // path.clear();

  while (node != nullptr) {

    float mean_q = node->get_mean_q(is_root, parent_q, this->discount);
    is_root = 0;
    parent_q = mean_q;

    int action = node->select_action(this->min_max_stats, this->pb_c_base,
                                     this->pb_c_init, this->discount, mean_q);
    last_node = node;
    last_action = action;
    node = node->get_child(action);

    // path.push_back(std::make_pair(last_node, action));
  }

  // std::cout<<"traverse path: ";
  // for(auto pr: path)
  // std::cout<< "(" << pr.first->id << ", " << pr.second <<") ";
  // std::cout<<std::endl;

  this->query_node = last_node;
  this->query_action = last_action;
  assert(this->query_node != nullptr);
  ret[0] = this->query_node->id;
  ret[1] = this->query_action;
  ret[2] = this->query_node->depth + 1; // search length
}

void Tree::back_propagate(float value_prefix, float value,
                          const std::vector<float> &policy, int is_reset,
                          const std::vector<int> &available_action) {
  assert(this->query_node != nullptr && this->query_action != -1);

  Node *node =
      new Node(this->num_nodes, this->query_node, value_prefix, policy);
  node->is_reset = is_reset;
  node->available_action.assign(available_action.begin(),
                                available_action.end());
  this->num_nodes++;
  this->node_list.push_back(node);
  this->query_node->explored_actions.push_back(this->query_action);
  this->query_node->children[this->query_action] = node;

  this->query_node = nullptr;
  this->query_action = -1;

  float bootstrap_value = value;
  for (; node != nullptr; node = node->parent) {
    node->value_sum += bootstrap_value;
    node->visit_count++;

    if (node->parent != nullptr) {
      Node *parent = node->parent;
      float true_reward =
          node->value_prefix - parent->value_prefix * (1 - parent->is_reset);
      bootstrap_value = true_reward + this->discount * bootstrap_value;
    }
  }
  this->min_max_stats.clear();
  this->update_tree_q(this->root);
}

std::vector<int> Tree::get_distribution() {
  return this->root->get_children_distribution();
}
float Tree::get_value() { return this->root->value(); }

void Tree::update_tree_q(Node *node) {
  if (node == nullptr)
    return;
  for (auto a : node->explored_actions) {
    Node *child = node->get_child(a);
    float true_reward =
        child->value_prefix - node->value_prefix * (1 - node->is_reset);
    float qsa = true_reward + this->discount * child->value();
    this->min_max_stats.update(qsa);

    this->update_tree_q(child);
  }
}

/*
    int num_trees, num_active_trees; // max number of trees
    std::vector<Tree*> tree_list;*/

Batch_MCTS::Batch_MCTS(int num_trees) {
  this->num_trees = num_trees;
  this->num_active_trees = 0;
  this->tree_list.clear();
  for (int i = 0; i < this->num_trees; i++)
    this->tree_list.push_back(new Tree());
}

Batch_MCTS::~Batch_MCTS() { this->clear(); }

void Batch_MCTS::reset(int pb_c_base, float pb_c_init, float discount,
                       float value_delta_max,
                       const std::vector<float> value_prefix,
                       const std::vector<std::vector<float>> policy,
                       const std::vector<std::vector<int>> available_action) {
  this->num_active_trees = value_prefix.size();
  for (int i = 0; i < this->num_active_trees; i++) {
    this->tree_list[i]->reset(pb_c_base, pb_c_init, discount, value_delta_max,
                              value_prefix[i], policy[i], available_action[i]);
  }
  for (int i = this->num_active_trees; i < this->num_trees; i++) {
    this->tree_list[i]->clear();
  }
  this->traverse_result.resize(this->num_active_trees);
  for (int i = 0; i < this->num_active_trees; i++)
    this->traverse_result[i].resize(3);
}

void Batch_MCTS::clear() {
  for (int i = 0; i < this->num_trees; i++)
    delete this->tree_list[i];
  this->tree_list.clear();
}

std::vector<std::vector<int>> Batch_MCTS::batch_traverse() {
  for (int i = 0; i < this->num_active_trees; i++) {
    this->tree_list[i]->traverse(this->traverse_result[i]);
  }
  return this->traverse_result;
}

void Batch_MCTS::batch_back_propagate(
    const std::vector<float> value_prefix, const std::vector<float> value,
    const std::vector<std::vector<float>> policy,
    const std::vector<int> is_reset_list,
    const std::vector<std::vector<int>> available_action) {
  for (int i = 0; i < this->num_active_trees; i++) {
    this->tree_list[i]->back_propagate(value_prefix[i], value[i], policy[i],
                                       is_reset_list[i], available_action[i]);
  }
}

std::vector<std::vector<int>> Batch_MCTS::batch_get_distribution() {
  std::vector<std::vector<int>> ret;
  ret.clear();
  for (int i = 0; i < this->num_active_trees; i++) {
    ret.push_back(this->tree_list[i]->get_distribution());
  }
  return ret;
}

std::vector<float> Batch_MCTS::batch_get_value() {
  std::vector<float> ret;
  ret.clear();
  for (int i = 0; i < this->num_active_trees; i++) {
    ret.push_back(this->tree_list[i]->get_value());
  }
  return ret;
}

void Batch_MCTS::print_all_trees() {
  for (int i = 0; i < this->num_active_trees; i++) {
    std::cout << "Tree " << i << std::endl;
    this->tree_list[i]->print(this->tree_list[i]->root);
  }
}

void Tree::print(Node *node) {
  if (node == nullptr)
    return;
  std::cout << "Node " << node->id << ":"
            << " Depth " << node->depth << ", Value_prefix "
            << node->value_prefix << ", Is_reset " << node->is_reset
            << std::endl;
  for (auto a : node->explored_actions) {
    auto child = node->get_child(a);
    std::cout << "    (Action " << a << ", Node " << child->id << ", Visit "
              << child->visit_count << ", Value " << child->value()
              << ", Value_prefix " << child->value_prefix << ") " << std::endl;
  }
  std::cout << std::endl;
  for (auto a : node->explored_actions) {
    auto child = node->get_child(a);
    this->print(child);
  }
}

Job::~Job() {}

Multithread_Batch_MCTS::Multithread_Batch_MCTS(int num_threads)
    : Batch_MCTS(0) {
  this->num_threads = num_threads;
  this->finish = false;
  this->reset_count = this->traverse_count = this->back_prop_count = 0;
  this->job_queue.clear();
  this->thread_pool.clear();

  for (int i = 0; i < num_threads; i++)
    this->thread_pool.emplace_back(
        std::thread(&Multithread_Batch_MCTS::worker, this, i));
}

Multithread_Batch_MCTS::~Multithread_Batch_MCTS() {

  this->finish = true;

  // std::cout<<"finish"<<std::endl;

  std::unique_lock<std::mutex> lck(this->mtx);
  this->job_queue.clear();
  this->worker_condvar.notify_all();
  lck.unlock();

  for (auto &th : this->thread_pool)
    th.join();

  this->clear();
}

void Multithread_Batch_MCTS::clear() {
  for (int i = 0; i < this->tree_list.size(); i++)
    delete this->tree_list[i];
  this->tree_list.clear();
  this->job_queue.clear();
  this->finish = false;

  // count
  this->reset_count = this->traverse_count = this->back_prop_count = 0;
  // traverse_result
  this->traverse_result.clear();
}

void Multithread_Batch_MCTS::reset(
    int pb_c_base, float pb_c_init, float discount, float value_delta_max,
    const std::vector<float> value_prefix,
    const std::vector<std::vector<float>> policy,
    const std::vector<std::vector<int>> available_action) {
  this->num_active_trees = value_prefix.size();

  // copy parameters
  this->reset_pb_c_base = pb_c_base;
  this->reset_pb_c_init = pb_c_init;
  this->reset_discount = discount;
  this->reset_value_delta_max = value_delta_max;
  this->reset_value_prefix.assign(value_prefix.begin(), value_prefix.end());
  this->reset_policy.assign(policy.begin(), policy.end());
  this->reset_available_action.assign(available_action.begin(),
                                      available_action.end());
  this->clear();
  this->traverse_result.resize(this->num_active_trees);
  for (int i = 0; i < this->num_active_trees; i++) {
    this->traverse_result[i].resize(3);
    this->tree_list.push_back(new Tree());
  }

  // std::cout<<"Try Reset"<<std::endl;

  // create new trees & push jobs
  {
    std::unique_lock<std::mutex> lck(this->mtx);
    // std::cout<<"Push Jobs"<<std::endl;
    for (int i = 0; i < this->num_active_trees; i++)
      this->job_queue.push_back(Job(i, JobType::Reset));
    this->worker_condvar.notify_all();
  }

  // wait for all trees to finish reset
  {
    std::unique_lock<std::mutex> lck(this->mtx);
    while (this->reset_count < this->num_active_trees)
      this->main_thread_condvar.wait(lck);
  }
}

std::vector<std::vector<int>> Multithread_Batch_MCTS::batch_traverse() {
  /*{
    std::unique_lock<std::mutex> lck(this->mtx);
    this->traverse_count = 0;
    for (int i = 0; i < this->num_active_trees; i++)
      this->job_queue.push_back(Job(i, JobType::Traverse));
    this->worker_condvar.notify_all();
  }

  {
    std::unique_lock<std::mutex> lck(this->mtx);
    while (this->traverse_count < this->num_active_trees)
      this->main_thread_condvar.wait(lck);
  }
  return this->traverse_result;*/

  for (int i = 0; i < this->num_active_trees; i++) {
    this->tree_list[i]->traverse(this->traverse_result[i]);
  }
  return this->traverse_result;
}

void Multithread_Batch_MCTS::batch_back_propagate(
    const std::vector<float> value_prefix, const std::vector<float> value,
    const std::vector<std::vector<float>> policy,
    const std::vector<int> is_reset_list,
    const std::vector<std::vector<int>> available_action) {
  this->back_prop_value_prefix.assign(value_prefix.begin(), value_prefix.end());
  this->back_prop_value.assign(value.begin(), value.end());
  this->back_prop_policy.assign(policy.begin(), policy.end());
  this->back_prop_is_reset_list.assign(is_reset_list.begin(),
                                       is_reset_list.end());
  this->back_prop_available_action.assign(available_action.begin(),
                                          available_action.end());

  {
    std::unique_lock<std::mutex> lck(this->mtx);
    this->back_prop_count = 0;
    for (int i = 0; i < this->num_active_trees; i++)
      this->job_queue.push_back(Job(i, JobType::Back_Propagate));
    this->worker_condvar.notify_all();
  }

  {
    std::unique_lock<std::mutex> lck(this->mtx);
    while (this->back_prop_count < this->num_active_trees)
      this->main_thread_condvar.wait(lck);
  }
}

void Multithread_Batch_MCTS::worker(int rank) {
  // std::cout<<"Start worker "<<rank<<std::endl;
  for (; !this->finish;) {
    std::unique_lock<std::mutex> lck(this->mtx);
    while (this->job_queue.empty() && !this->finish) {
      this->main_thread_condvar.notify_all();
      this->worker_condvar.wait(lck);
    }

    if (this->finish)
      break;

    Job j = this->job_queue.back();
    this->job_queue.pop_back();
    lck.unlock();
    std::vector<int> ret;
    // std::cout<<"Workr "<<rank<<" get job "<<j.tree_idx<<"
    // "<<j.job_type<<std::endl;
    switch (j.job_type) {
    case JobType::Reset:
      this->tree_list[j.tree_idx]->reset(
          this->reset_pb_c_base, this->reset_pb_c_init, this->reset_discount,
          this->reset_value_delta_max, this->reset_value_prefix[j.tree_idx],
          this->reset_policy[j.tree_idx],
          this->reset_available_action[j.tree_idx]);
      this->reset_count++;
      break;
    case JobType::Traverse:
      this->tree_list[j.tree_idx]->traverse(this->traverse_result[j.tree_idx]);
      this->traverse_count++;
      break;
    case JobType::Back_Propagate:
      this->tree_list[j.tree_idx]->back_propagate(
          this->back_prop_value_prefix[j.tree_idx],
          this->back_prop_value[j.tree_idx], this->back_prop_policy[j.tree_idx],
          this->back_prop_is_reset_list[j.tree_idx],
          this->back_prop_available_action[j.tree_idx]);
      this->back_prop_count++;
      break;
    default:
      throw std::runtime_error("Not implemented job encountered!");
    }
  }
}
} // namespace mcts

/*
TODO:
1. pybind
2. test tree structure
3. check the code again
*/

namespace py = pybind11;

PYBIND11_MODULE(batch_mcts, m) {
  py::class_<mcts::Batch_MCTS>(m, "Batch_MCTS")
      .def(py::init<int>())
      .def("reset", &mcts::Batch_MCTS::reset)
      .def("clear", &mcts::Batch_MCTS::clear)
      .def("batch_traverse", &mcts::Batch_MCTS::batch_traverse)
      .def("batch_back_propagate", &mcts::Batch_MCTS::batch_back_propagate)
      .def("batch_get_distribution", &mcts::Batch_MCTS::batch_get_distribution)
      .def("batch_get_value", &mcts::Batch_MCTS::batch_get_value)
      .def("print_all_trees", &mcts::Batch_MCTS::print_all_trees);

  py::class_<mcts::Multithread_Batch_MCTS>(m, "Multithread_Batch_MCTS")
      .def(py::init<int>())
      .def("reset", &mcts::Multithread_Batch_MCTS::reset)
      .def("clear", &mcts::Multithread_Batch_MCTS::clear)
      .def("batch_traverse", &mcts::Multithread_Batch_MCTS::batch_traverse)
      .def("batch_back_propagate",
           &mcts::Multithread_Batch_MCTS::batch_back_propagate)
      .def("batch_get_distribution",
           &mcts::Multithread_Batch_MCTS::batch_get_distribution)
      .def("batch_get_value", &mcts::Multithread_Batch_MCTS::batch_get_value)
      .def("print_all_trees", &mcts::Multithread_Batch_MCTS::print_all_trees);
}
