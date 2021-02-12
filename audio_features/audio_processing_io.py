import ffmpeg
import os


def extract_scene_audio(film, scene_dict):
    input_audio_file = os.path.join('../input_audio/' + film + '.wav')
    input_stream = ffmpeg.input(input_audio_file)
    first = str(scene_dict['first_frame'])
    last = str(scene_dict['last_frame'])

    extracted_file_name = os.path.join('../extracted_audio', film, first + '_' + last + '.wav')
    out = ffmpeg.output(input_stream, extracted_file_name, ss=scene_dict['first_frame'], ac=2, t=scene_dict['scene_duration'])
    ffmpeg.run(out, overwrite_output=True)

    print('Extracted audio file:', extracted_file_name)
