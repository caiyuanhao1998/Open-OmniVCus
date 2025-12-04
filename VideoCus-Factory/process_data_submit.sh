project=ilo-train-p4de
jack_id=5862
name_runai=sam-data-all
USER=xic


runai submit \
--name ${name_runai} \
--large-shm \
-i docker-matrix-experiments-snapshot.dr-uw2.adobeitc.com/dit-cuda12.4-pytorch2.3:v4 \
-g 8 \
-p ${project} \
-l research_jack_id=${jack_id} \
-l activity_type=development \
-e USER=$USER \
--node-pools a100-80gb-1 \
--service-type=external-url --port 8888 \
--command -- /bin/bash -c \ "/sensei-fs/users/xic/project/codebase/segment-anything-2/process_data.sh"