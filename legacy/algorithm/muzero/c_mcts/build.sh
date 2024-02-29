BASEDIR=$(dirname "$0")

if [[ $OSTYPE == 'darwin'* ]]; then
  echo 'Building batch mcts on macOS'
  c++ -O3 -Wall -shared -std=c++11 -Wsign-compare -fPIC -undefined dynamic_lookup $(python3 -m pybind11 --includes) $BASEDIR/ctree.cc -o $BASEDIR/batch_mcts$(python3-config --extension-suffix)
elif [[ $OSTYPE == 'linux'* ]]; then
  echo 'Building batch mcts on linux'
  c++ -O3 -Wall -shared -std=c++11 -Wsign-compare -fPIC $(python3 -m pybind11 --includes) $BASEDIR/ctree.cc $BASEDIR/ctree.h -o $BASEDIR/batch_mcts$(python3-config --extension-suffix)
fi
