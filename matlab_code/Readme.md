**Matlabcode**

** DONOT USE patch16800 until further NOTICE**


* General Rules
	* DO NOT change values other than those in  acustomize section
	* Make sure that /Data/Mask file is the same on from /Personal_folder/woojin/QSM_challenge/QSMnet_code&data/Mask/
	* Usually, 17640 means it fetches data from the patient #1 to #7
	* Usually, 16800 means it fetches data from the patient #1 to #5


* What each title means?
	*patch17640\_linearity\_50percent
		*A half of patch data is constructed using dipole convolution. You can change its scale factor by tweaking lambda
	*patch17640\_linearity\_25\_25\_percent
		*A half of patch data is constructed using dipole convolution. But a half of the half(a quarter in total) will be scaled according to lambda_1 and another half of the half(another quarter in total) will be scaled according to labmda_2.

* Customize Section
	* DATA_NAME: Make sure you include upto .mat Matlab will save data at the location specified by this value
	* training\_before..: Put in upto training\_before\_patches.mat
	* patientSet: usually upto 5 or 7. You CAN customize it, but make sure that you have corresponding mask data at /Data/Mask. As long as you play around in one to seven, you are good.
	* lambda\_1, labmda\_2: linearity factor. 2 and 4 are default values.


