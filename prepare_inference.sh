if [ ! ${PWD##*/} = "Data" ]; then
    echo "Execute this script in the Data folder!"
else
	nas_base="$1/Personal_folder/woojin/QSM_challenge/Data"
    if [ ! -f "Challenge_stage2_sim1snr1_VSHARP.mat" ]; then
		cp -v "$nas_base/Challenge_stage2_sim1snr1_VSHARP.mat" ./
	fi
	if [ ! -f "Challenge_stage2_sim1snr2_VSHARP.mat" ]; then
		cp -v "$nas_base/Challenge_stage2_sim1snr2_VSHARP.mat" ./
	fi
	if [ ! -f "Challenge_stage2_sim2snr1_VSHARP.mat" ]; then
		cp -v "$nas_base/Challenge_stage2_sim2snr1_VSHARP.mat" ./
	fi
	if [ ! -f "Challenge_stage2_sim2snr2_VSHARP.mat" ]; then
		cp -v "$nas_base/Challenge_stage2_sim2snr2_VSHARP.mat" ./
	fi
	if [ ! -f "Challenge_stage2_sim1snr1_PDF.mat" ]; then
		cp -v "$nas_base/Challenge_stage2_sim1snr1_PDF.mat" ./
	fi
	if [ ! -f "Challenge_stage2_sim1snr2_PDF.mat" ]; then
		cp -v "$nas_base/Challenge_stage2_sim1snr2_PDF.mat" ./
	fi
	if [ ! -f "Challenge_stage2_sim2snr1_PDF.mat" ]; then
		cp -v "$nas_base/Challenge_stage2_sim2snr1_PDF.mat" ./
	fi
	if [ ! -f "Challenge_stage2_sim2snr2_PDF.mat" ]; then
		cp -v "$nas_base/Challenge_stage2_sim2snr2_PDF.mat" ./
	fi
	if [ ! -f "Challenge_stage1_ROI_mask.mat" ]; then
		cp -v "$nas_base/Challenge_stage1_ROI_mask.mat" ./
	fi
	if [ ! -f "Test_0718.mat" ]; then
		cp -v "$nas_base/Test_0718.mat" ./
	fi
	if [ ! -f "Test_ROI_mask.mat" ]; then
		cp -v "$nas_base/Test_ROI_mask.mat" ./
	fi

	nas_base="$nas_base/Conventional"
	if [ ! -d "Conventional" ]; then
		mkdir -v Conventional
	fi
	if [ ! -f "Conventional/Sim1_iLSQR_fix.mat" ]; then
		cp -v "$nas_base/Sim1_iLSQR_fix.mat" Conventional
	fi
	if [ ! -f "Conventional/Sim1_MEDI_fix.mat" ]; then
       	cp -v "$nas_base/Sim1_MEDI_fix.mat" Conventional
    fi
	if [ ! -f "Conventional/Sim1_QSMNet_fix.mat" ]; then
		cp -v "$nas_base/Sim1_QSMNet_fix.mat" Conventional
	fi
	if [ ! -f "Conventional/Sim2_iLSQR_fix.mat" ]; then
		cp -v "$nas_base/Sim2_iLSQR_fix.mat" Conventional
	fi
	if [ ! -f "Conventional/Sim2_MEDI_fix.mat" ]; then
		cp -v "$nas_base/Sim2_MEDI_fix.mat" Conventional
	fi
	if [ ! -f "Conventional/Sim2_QSMNet_fix.mat" ]; then
		cp -v "$nas_base/Sim2_QSMNet_fix.mat" Conventional
	fi
	if [ ! -f "Conventional/Test_iLSQR.mat" ]; then
		cp -v "$nas_base/Test_iLSQR.mat" Conventional
	fi
	if [ ! -f "Conventional/Test_StarQSM.mat" ]; then
		cp -v "$nas_base/Test_StarQSM.mat" Conventional
	fi
	if [ ! -f "Conventional/Test_QSMNet.mat" ]; then
		cp -v "$nas_base/Test_QSMNet.mat" Conventional
	fi

	echo "Done."
fi	
