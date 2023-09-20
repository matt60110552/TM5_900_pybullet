
#!/usr/bin/env bash

BASH_OPTION=bash

IMG=iscilab/rl_matt:cuda-20-04
containerid=$(docker ps -qf "ancestor=${IMG}") && echo $containerid

xhost +

if [[ -n "$containerid" ]]
then
    docker exec -it \
        --privileged \
        -e DISPLAY=${DISPLAY} \
        -e LINES="$(tput lines)" \
        rl_matt \
        $BASH_OPTION
else
    docker start -i rl_matt
fi
