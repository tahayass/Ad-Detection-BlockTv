import csv
from datetime import datetime, timedelta

import csv
from datetime import datetime, timedelta

def parse_segments_from_txt(txt_filename):
    with open(txt_filename, 'r') as txt_file:
        lines = txt_file.readlines()

    segments = []
    for i in range(0, len(lines), 2):
        start_time_str = lines[i].strip().split('=')[1]
        end_time_str = lines[i + 1].strip().split('=')[1]

        start_time_parts = list(map(int, start_time_str.split(':')))
        end_time_parts = list(map(int, end_time_str.split(':')))

        start_time_seconds = start_time_parts[0] * 3600 + start_time_parts[1] * 60 + start_time_parts[2]
        end_time_seconds = end_time_parts[0] * 3600 + end_time_parts[1] * 60 + end_time_parts[2]

        segment_length = end_time_seconds - start_time_seconds
        rename_to = f"video{i//2 + 1}"

        segments.append({
            'start_time': start_time_seconds,
            'length': segment_length,
            'rename_to': rename_to
        })

    # Add intermediate segments
    for i in range(len(segments) - 1):
        current_end_time = segments[i]['start_time'] + segments[i]['length']
        next_start_time = segments[i + 1]['start_time']

        if current_end_time < next_start_time:
            intermediate_length = min(20 * 60, next_start_time - current_end_time)

            segments.append({
                'start_time': current_end_time,
                'length': intermediate_length,
                'rename_to': f"{segments[i]['rename_to']}_intermediate"
            })

    return segments

def write_csv(segments, csv_filename):
    with open(csv_filename, mode='w', newline='') as csv_file:
        fieldnames = ['start_time', 'length', 'rename_to']
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)

        writer.writeheader()
        for segment in segments:
            writer.writerow(segment)

if __name__ == "__main__":
    total_duration = (24 * 60 + 48) * 60  # 24 hours and 48 minutes in seconds
    segment_duration = 20 * 60  # 20 minutes in seconds
    csv_filename = 'video_segments.csv'
    txt_path=r'generate_dataset\adds-timestamps_test.txt'

    segments = parse_segments_from_txt(txt_path)
    write_csv(segments, csv_filename)

    print(f"CSV file '{csv_filename}' created successfully.")
