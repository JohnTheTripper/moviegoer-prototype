import datetime


def frame_to_time(frame_number):
    seconds = frame_number % (24 * 3600)
    hours = seconds // 3600
    seconds %= 3600
    minutes = seconds // 60
    seconds %= 60

    timestamp = datetime.time(hours, minutes, seconds)

    return timestamp


def time_to_frame(time):
    frame_number = (time.hour * 3600) + (time.minute * 60) + (time.second)
    if time.microsecond >= 500000:
        frame_number += 1

    return frame_number


def subtitle_mid_time(start_time, end_time):
    delta = datetime.datetime.combine(datetime.date.today(), end_time) - datetime.datetime.combine(
        datetime.date.today(), start_time)
    mid_time_day = (delta / 2) + datetime.datetime.combine(datetime.date.today(), start_time)
    mid_time = datetime.time(mid_time_day.hour, mid_time_day.minute, mid_time_day.second, mid_time_day.microsecond)

    return mid_time


def add_time_offset(time_object, offset):
    datetime_object = datetime.datetime(year=2000, month=1, day=1, hour=time_object.hour, minute=time_object.minute,
                                        second=time_object.second, microsecond=time_object.microsecond)
    datetime_offset = datetime.timedelta(milliseconds=offset)
    datetime_set_off = datetime_object + datetime_offset
    time_added = datetime.time(hour=datetime_set_off.hour, minute=datetime_set_off.minute,
                               second=datetime_set_off.second, microsecond=datetime_set_off.microsecond)
    return time_added


def subtract_time_offset(time_object, offset):
    datetime_object = datetime.datetime(year=2000, month=1, day=1, hour=time_object.hour, minute=time_object.minute,
                                        second=time_object.second, microsecond=time_object.microsecond)
    datetime_offset = datetime.timedelta(milliseconds=offset)
    datetime_set_off = datetime_object - datetime_offset
    time_subtracted = datetime.time(hour=datetime_set_off.hour, minute=datetime_set_off.minute,
                                    second=datetime_set_off.second, microsecond=datetime_set_off.microsecond)
    return time_subtracted


def subtract_time_objects(later_time, earlier_time):
    earlier_datetime_object = datetime.datetime(year=2000, month=1, day=1, hour=earlier_time.hour,
                                                minute=earlier_time.minute,
                                                second=earlier_time.second, microsecond=earlier_time.microsecond)
    later_datetime_object = datetime.datetime(year=2000, month=1, day=1, hour=later_time.hour, minute=later_time.minute,
                                              second=later_time.second, microsecond=later_time.microsecond)

    time_difference = later_datetime_object - earlier_datetime_object

    return time_difference
