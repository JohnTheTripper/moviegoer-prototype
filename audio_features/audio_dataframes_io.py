import pandas as pd
from conversations_io import *
import sys
sys.path.append('../unifying_features')
from time_reference_io import *

def generate_conversation_df(audio_file, frame_choice):
    audible_sound = analyze_audible_sound(audio_file, plot=False)
    speaker_list = cluster_voices(audio_file, audible_sound, plot=False)

    times = []
    for frame in frame_choice:
        times.append(frame_to_time(frame))

    zeroed_times = []
    for frame in range(0, len(frame_choice)):
        zeroed_times.append(frame_to_time(frame))

    conversation_df = pd.DataFrame(zip(frame_choice, times, zeroed_times, speaker_list, audible_sound),
                                   columns=['frame', 'time', 'zeroed_time', 'speaker', 'audible_sound'])
    conversation_df = conversation_df.set_index('frame')

    return conversation_df