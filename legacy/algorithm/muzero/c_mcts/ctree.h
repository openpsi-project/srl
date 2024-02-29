#include <atomic>
#include <cmath>
#include <condition_variable>
#include <math.h>
#include <mutex>
#include <stack>
#include <stdlib.h>
#include <thread>
#include <vector>

const float FLOAT_MAX = 1000000.0;
const float FLOAT_MIN = -FLOAT_MAX;

namespace tools {

class MinMaxStats {
public:
  float maximum, minimum, value_delta_max;

  MinMaxStats();
  ~MinMaxStats();

  void set_delta(float value_delta_max);
  void update(float value);
  void clear();
  float normalize(float value);
};
} // namespace tools

namespace mcts {
class Node {
public:
  // MuZero related
  int visit_count, action_num, is_reset;
  float value_prefix, value_sum;
  std::vector<int> available_action;

  // tree structure
  int id, depth;
  Node *parent;
  std::vector<Node *> children;
  std::vector<int> explored_actions;
  std::vector<float> prior;

  Node(int id, Node *parent, float value_prefix,
       const std::vector<float> &policy);
  ~Node();

  float get_mean_q(int isRoot, float parent_q, float discount);
  float value();
  Node *get_child(int action);
  std::vector<int> get_children_distribution();
  int select_action(tools::MinMaxStats &min_max_stats, int pb_c_base,
                    float pb_c_init, float discount, float mean_q);
};

class Tree {
public:
  // MuZero related
  int pb_c_base;
  float pb_c_init, discount;
  tools::MinMaxStats min_max_stats;

  // tree structure
  Node *root;
  std::vector<Node *> node_list;
  int num_nodes;

  // MCTS
  Node *query_node;
  int query_action;

  Tree();
  ~Tree();

  void clear();
  void reset(int pb_c_base, float pb_c_init, float discount,
             float value_delta_max, float value_prefix,
             const std::vector<float> &policy,
             const std::vector<int> &available_action);

  void traverse(std::vector<int> &ret);
  void back_propagate(float value_prefix, float value,
                      const std::vector<float> &policy, int is_reset,
                      const std::vector<int> &available_action);
  std::vector<int> get_distribution();
  float get_value();
  void update_tree_q(Node *node);

  void print(Node *node);
};

class Batch_MCTS {
public:
  int num_trees, num_active_trees; // max number of trees
  std::vector<Tree *> tree_list;
  std::vector<std::vector<int>> traverse_result;

  Batch_MCTS(int num_trees);
  ~Batch_MCTS();

  virtual void reset(int pb_c_base, float pb_c_init, float discount,
                     float value_delta_max,
                     const std::vector<float> value_prefix,
                     const std::vector<std::vector<float>> policy,
                     const std::vector<std::vector<int>> available_action);
  virtual void clear();

  virtual std::vector<std::vector<int>> batch_traverse();
  virtual void
  batch_back_propagate(const std::vector<float> value_prefix,
                       const std::vector<float> value,
                       const std::vector<std::vector<float>> policy,
                       const std::vector<int> is_reset_list,
                       const std::vector<std::vector<int>> available_action);
  std::vector<std::vector<int>> batch_get_distribution();
  std::vector<float> batch_get_value();

  void print_all_trees();
};

enum JobType { Reset = 0, Traverse, Back_Propagate, No_Job };

class Job {
public:
  int tree_idx;
  JobType job_type;

  Job(int _tree_idx = -1, JobType _job_type = JobType::No_Job)
      : tree_idx(_tree_idx), job_type(_job_type) {}
  ~Job();
};

class Multithread_Batch_MCTS : public Batch_MCTS {
public:
  void reset(int pb_c_base, float pb_c_init, float discount,
             float value_delta_max, const std::vector<float> value_prefix,
             const std::vector<std::vector<float>> policy,
             const std::vector<std::vector<int>> available_action) override;
  void clear() override;

  std::vector<std::vector<int>> batch_traverse() override;
  void batch_back_propagate(
      const std::vector<float> value_prefix, const std::vector<float> value,
      const std::vector<std::vector<float>> policy,
      const std::vector<int> is_reset_list,
      const std::vector<std::vector<int>> available_action) override;

  // public:
  Multithread_Batch_MCTS(int num_threads);
  ~Multithread_Batch_MCTS();

private:
  int num_threads;
  std::vector<std::thread> thread_pool;
  std::condition_variable worker_condvar, main_thread_condvar;
  std::mutex mtx;
  std::vector<Job> job_queue;
  std::atomic<bool> finish;

  // reset
  int reset_pb_c_base;
  float reset_pb_c_init, reset_discount, reset_value_delta_max;
  std::vector<float> reset_value_prefix;
  std::vector<std::vector<float>> reset_policy;
  std::vector<std::vector<int>> reset_available_action;
  std::atomic<int> reset_count;
  // traverse
  std::atomic<int> traverse_count;
  // back_prop
  std::vector<float> back_prop_value_prefix, back_prop_value;
  std::vector<std::vector<float>> back_prop_policy;
  std::vector<int> back_prop_is_reset_list;
  std::vector<std::vector<int>> back_prop_available_action;
  std::atomic<int> back_prop_count;

  void worker(int rank);
};
} // namespace mcts