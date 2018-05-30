function ask_owerwrite () {
  local exist=0;
  local expect="$#";
  local files=("$@");
  while [ $# -gt 0 ]; do
    if [ -f "$1" ]; then ((++exist)); fi;
    shift 1;
  done;
  if [ "$exist" -eq "$expect" ]; then
    msg="Files \"${files[@]}\" already exist. Overwrite them (y or n)? ";
    read -p "$msg" -n 1 -r; echo;
    if [[ $REPLY =~ ^[Yy]$ ]]; then
      return 1;
    else
      return 0;
    fi;
  fi;
  return 1;
}

function memusg_wrap () {
  if which memusg &> /dev/null; then
    memusg "$@";
  else
    "$@";
  fi;
}
