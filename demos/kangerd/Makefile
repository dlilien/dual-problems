volumes.pdf contours.pdf &: make_plots.py kangerdlugssuaq-year5.h5
	python make_plots.py

kangerdlugssuaq-year5.h5: simulate.py kangerdlugssuaq-year1.h5
	python simulate.py \
		--input kangerdlugssuaq-year1.h5 \
		--final-time 4.0 \
		--timesteps-per-year 96 \
		# --calving \
		--melt-rate 28e3 \
		--mask-smoothing-length 1e3 \
		--snes-max-it 500 \
		--snes-rtol 1e-6 \
		--output kangerdlugssuaq-year5.h5

kangerdlugssuaq-year1.h5: simulate.py kangerdlugssuaq-extrapolated.h5
	python simulate.py \
		--input kangerdlugssuaq-extrapolated.h5 \
		--final-time 1.0 \
		--timesteps-per-year 192 \
		--snes-rtol 1e-6 \
		--output kangerdlugssuaq-year1.h5

kangerdlugssuaq-year1-3d.h5: simulate_3d.py kangerdlugssuaq-extrapolated.h5
	python simulate_3d.py \
		--input kangerdlugssuaq-extrapolated.h5 \
		--final-time 1.0 \
		--timesteps-per-year 384 \
		--snes-rtol 1e-6 \
		--output kangerdlugssuaq-year1-3d.h5

kangerdlugssuaq-extrapolated.h5: extrapolate.py kangerdlugssuaq-initial.h5
	python extrapolate.py

kangerdlugssuaq-initial.h5: initialize.py
	python initialize.py

kangerdlugssuaq-year1-bp.h5: simulate_bp.py kangerdlugssuaq-extrapolated.h5
	python simulate_bp.py \
		--input kangerdlugssuaq-extrapolated-bp.h5 \
		--final-time 1.0 \
		--timesteps-per-year 384 \
		--snes-rtol 1e-6 \
		--output kangerdlugssuaq-year1-bp.h5