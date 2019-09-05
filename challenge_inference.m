function ReconMetrics = challenge_inference(recon_base_path, nas_base_path, sim, snr)

ReconName = [recon_base_path, '/_Sim', int2str(sim), 'Snr', int2str(snr), '_Step2.nii'];
GroundTruth = load([nas_base_path, '/Sim', int2str(sim), 'Snr', int2str(snr), '/GT/FilestructureForEval.mat'],'filesstructure');

GroundTruth.filesstructure.Segment = [nas_base_path, '/', GroundTruth.filesstructure.Segment];
GroundTruth.filesstructure.maskEroded = [nas_base_path, '/', GroundTruth.filesstructure.maskEroded];
GroundTruth.filesstructure.label = [nas_base_path, '/', GroundTruth.filesstructure.label];
GroundTruth.filesstructure.chi_crop = [nas_base_path, '/', GroundTruth.filesstructure.chi_crop];

ReconMetrics = EvaluateRecon_ChallengeFinalMetrics(ReconName, GroundTruth.filesstructure);