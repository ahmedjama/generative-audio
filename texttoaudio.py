from transformers import pipeline
import scipy

synthesiser = pipeline("text-to-audio", "facebook/musicgen-medium")

music = synthesiser("traditional east african beat", forward_params={"do_sample": True})

scipy.io.wavfile.write("musicgen_eaf_out.wav", rate=music["sampling_rate"], data=music["audio"])
