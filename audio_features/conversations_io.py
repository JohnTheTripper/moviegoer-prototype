import pyAudioAnalysis.audioSegmentation
import numpy as np


def analyze_audible_sound(audio_file, plot=False):
    sampling_rate, signal = pyAudioAnalysis.audioBasicIO.read_audio_file(audio_file)
    segments_with_sound = pyAudioAnalysis.audioSegmentation.silence_removal(signal, sampling_rate, st_win=0.05,
                                                                            st_step=0.025, smooth_window=0.5,
                                                                            weight=0.3, plot=plot)
    audible_sound = []

    for frame in range(0, int(len(signal) / sampling_rate)):
        sound_found = 0

        for segment in segments_with_sound:
            sound_start = segment[0]
            sound_end = segment[1]
            if sound_start <= frame <= sound_end:
                sound_found = 1
            if sound_start <= frame + .25 <= sound_end:
                sound_found = 1
            if sound_start <= frame + .5 <= sound_end:
                sound_found = 1
            if sound_start <= frame + .75 <= sound_end:
                sound_found = 1
            if sound_start <= frame + .999 <= sound_end:
                sound_found = 1

        if sound_found:
            audible_sound.append(1)
        else:
            audible_sound.append(0)

    return audible_sound


def cluster_voices(audio_file, audible_sound, plot=True):
    clusters = pyAudioAnalysis.audioSegmentation.speaker_diarization(audio_file, n_speakers=2, mid_window=0.8,
                                                                     mid_step=0.1, short_window=0.02, lda_dim=0,
                                                                     plot_res=plot)

    speaker_list = []
    x = 0
    while x < len(clusters):
        if sum(clusters[x:x + 10]) == 5 or audible_sound[int(x / 10)] == 0:  # ignore tie
            speaker_list.append(0)
        else:
            speaker_list.append(chr(int(round(np.mean(clusters[x:x + 10]))) + 77))  # convert 0 and 1 to M and N
        x += 10

    return speaker_list


