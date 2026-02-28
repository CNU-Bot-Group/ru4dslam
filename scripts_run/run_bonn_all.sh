#!/bin/bash

python run.py ./configs/Dynamic/Bonn/bonn_balloon.yaml
python gen_rum.py ./configs/Dynamic/Bonn/bonn_balloon.yaml
python precompute_4dgs.py --cfg ./configs/Dynamic/Bonn/bonn_balloon.yaml --ws test
python reconstruct_4dgs.py --cfg ./configs/Dynamic/Bonn/bonn_balloon.yaml --ws test

python run.py ./configs/Dynamic/Bonn/bonn_balloon2.yaml
python gen_rum.py ./configs/Dynamic/Bonn/bonn_balloon2.yaml
python precompute_4dgs.py --cfg ./configs/Dynamic/Bonn/bonn_balloon2.yaml --ws test
python reconstruct_4dgs.py --cfg ./configs/Dynamic/Bonn/bonn_balloon2.yaml --ws test

python run.py ./configs/Dynamic/Bonn/bonn_person_tracking.yaml
python gen_rum.py ./configs/Dynamic/Bonn/bonn_person_tracking.yaml
python precompute_4dgs.py --cfg ./configs/Dynamic/Bonn/bonn_person_tracking.yaml --ws test
python reconstruct_4dgs.py --cfg ./configs/Dynamic/Bonn/bonn_person_tracking.yaml --ws test

python run.py ./configs/Dynamic/Bonn/bonn_person_tracking2.yaml
python gen_rum.py ./configs/Dynamic/Bonn/bonn_person_tracking2.yaml
python precompute_4dgs.py --cfg ./configs/Dynamic/Bonn/bonn_person_tracking2.yaml --ws test
python reconstruct_4dgs.py --cfg ./configs/Dynamic/Bonn/bonn_person_tracking2.yaml --ws test

python run.py ./configs/Dynamic/Bonn/bonn_synchronous.yaml
python gen_rum.py ./configs/Dynamic/Bonn/bonn_synchronous.yaml
python precompute_4dgs.py --cfg ./configs/Dynamic/Bonn/bonn_synchronous.yaml --ws test
python reconstruct_4dgs.py --cfg ./configs/Dynamic/Bonn/bonn_synchronous.yaml --ws test

python run.py ./configs/Dynamic/Bonn/bonn_synchronous2.yaml
python gen_rum.py ./configs/Dynamic/Bonn/bonn_synchronous2.yaml
python precompute_4dgs.py --cfg ./configs/Dynamic/Bonn/bonn_synchronous2.yaml --ws test
python reconstruct_4dgs.py --cfg ./configs/Dynamic/Bonn/bonn_synchronous2.yaml --ws test

python run.py ./configs/Dynamic/Bonn/bonn_person_placing_nonobstructing_box.yaml
python gen_rum.py ./configs/Dynamic/Bonn/bonn_person_placing_nonobstructing_box.yaml
python precompute_4dgs.py --cfg ./configs/Dynamic/Bonn/bonn_person_placing_nonobstructing_box.yaml --ws test
python reconstruct_4dgs.py --cfg ./configs/Dynamic/Bonn/bonn_person_placing_nonobstructing_box.yaml --ws test

python run.py ./configs/Dynamic/Bonn/bonn_person_placing_nonobstructing_box2.yaml
python gen_rum.py ./configs/Dynamic/Bonn/bonn_person_placing_nonobstructing_box2.yaml
python precompute_4dgs.py --cfg ./configs/Dynamic/Bonn/bonn_person_placing_nonobstructing_box2.yaml --ws test
python reconstruct_4dgs.py --cfg ./configs/Dynamic/Bonn/bonn_person_placing_nonobstructing_box2.yaml --ws test

python run.py ./configs/Dynamic/Bonn/bonn_person_placing_nonobstructing_box3.yaml
python gen_rum.py ./configs/Dynamic/Bonn/bonn_person_placing_nonobstructing_box3.yaml
python precompute_4dgs.py --cfg ./configs/Dynamic/Bonn/bonn_person_placing_nonobstructing_box3.yaml --ws test
python reconstruct_4dgs.py --cfg ./configs/Dynamic/Bonn/bonn_person_placing_nonobstructing_box3.yaml --ws test


