gibbous.pdf volumes.pdf counts.pdf &: calving.h5 primal-counts.json dual-counts.json make_plots.py
	python make_plots.py --calving-freq 24.0

calving.h5: steady-state-fine.h5 gibbous.py
	python gibbous.py --resolution 2e3 --final-time 200.0 --num-steps 400 --calving-freq 24.0 --input $< --output $@

steady-state-fine.h5: steady-state-coarse.h5 gibbous.py
	python gibbous.py --resolution 2e3 --final-time 400.0 --num-steps 400 --input $< --output $@

%-counts.json: steady-state-coarse.h5 primal_dual_calving.py
	python primal_dual_calving.py --form $* --hmin 1e-3 --final-time 60 --num-steps 60 --calving-frequency 24 --output $@

steady-state-coarse.h5: gibbous.py
	python gibbous.py --resolution 5e3 --final-time 400.0 --num-steps 200 --output $@

gibbous.pdf volumes.pdf counts.pdf &: calving.h5 primal-counts.json dual-counts.json make_plots.py
	python make_plots.py --calving-freq 24.0

calving.h5: steady-state-fine.h5 gibbous.py
	python gibbous.py --resolution 2e3 --final-time 200.0 --num-steps 400 --calving-freq 24.0 --input $< --output $@

steady-state-fine-bp.h5: steady-state-coarse-bp.h5 gibbous-bp.py
	python gibbous-bp.py --resolution 2e3 --final-time 400.0 --num-steps 400 --input $< --output $@

%-counts.json: steady-state-coarse.h5 primal_dual_calving.py
	python primal_dual_calving.py --form $* --hmin 1e-3 --final-time 60 --num-steps 60 --calving-frequency 24 --output $@

steady-state-coarse-bp.h5: gibbous-bp.py
	python gibbous-bp.py --resolution 5e3 --final-time 400.0 --num-steps 200 --output $@

steady-state-fine-3d.h5: steady-state-coarse-3d.h5 gibbous-3d.py
	python gibbous-3d.py --resolution 2e3 --final-time 400.0 --num-steps 400 --input $< --output $@

steady-state-coarse-3d.h5: gibbous-3d.py
	python gibbous-3d.py --resolution 5e3 --final-time 400.0 --num-steps 200 --output $@