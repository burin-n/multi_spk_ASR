import os
import random
import wave
import contextlib
import json

def get_global_spk_info(spk_file):
	spk_info = {}
	with open(spk_file) as f:
		for line in f:
			if line[0] == ';': continue
			tokens = [x.strip(' ')for x in line.strip().split('|')]
			spk_id, sex, subset, minutes, name = tokens[0], tokens[1], tokens[2], tokens[3], ' '.join(tokens[4:])
			spk_info[spk_id] = {
				'sex' : sex.lower(),
				'subset' : subset,
				'minutes' : minutes,
				'name' : name,
			}
	return spk_info


def random_delays(wav_durations, min_delay=0.5, min_overlap=1):
	# delay at least 0.5 seconds
	# guarantee at least one overlap region
	delays = [0.0]
	total_dur = wav_durations[0]
	min_start = min_delay
	for dur in wav_durations[1:]:
		min_overlap_t = min(min_overlap, total_dur-min_start)
		delay = random.uniform(min_start, total_dur-min_overlap_t)
		total_dur = max( total_dur, dur+delay )
		min_start = min_delay + delay
		delays.append(delay)

	total_dur = wav_durations[0]
	# print(delays)
	# print(wav_durations)
	for i in range(1, len(delays)):
		min_overlap_t = min(min_overlap, total_dur-min_start)
		assert delays[i] <= total_dur - min_overlap_t
		assert delays[i-1] <= delays[i] - min_delay
		total_dur += wav_durations[i]

	return delays


def random_spk(spk_list, global_spk_info=None, n=2):
	"""
		spk_list: a list contains id of speakers
		global_spk_info: for gender 
		n: a number of speakers to be picked
	"""
	spks = []
	while len(spks) < n:
		pick = random.randint(0, len(spk_list)-1)
		if pick not in spks:
			spks.append(pick)
	
	if(global_spk_info == None):
		return [spk_list[picked_spk] for picked_spk in spks]
	else:
		spk_list = [spk_list[picked_spk] for picked_spk in spks]
		gender_list = [global_spk_info[picked_spk]["sex"] for picked_spk in spk_list]
		return spk_list, gender_list

def random_utterance(spk_list, data_dict, n=1):
	"""
		spk_list: a list contains id of speakers
		n: a lists contains number of utterances for each speakers
	"""
	if ( n > 1 ): raise NotImplementedError()
	wavs = []
	durs = []
	for spk in spk_list:
		rand_wav = random.randint(0, len(data_dict[spk])-1)
		wavs.append(data_dict[spk]["wavs"][rand_wav])
		durs.append(data_dict[spk]["durs"][rand_wav])
	return wavs, durs


def get_transcript(transcript, wav_path=None, utt_id=None):
	if(wav_path == None and utt_id == None):
		raise ValueError("Either wav_path or utt_id must not be None")
	if(wav_path != None and utt_id != None):
		raise ValueError("Only one of either wav_path or utt_id can be specify")
	if(wav_path != None):
		utt_id = [os.path.basename(wav)[:-4] for wav in wav_path]
	return [transcript[utt] for utt in utt_id]


def get_duration(fname):
	with contextlib.closing(wave.open(fname,'r')) as f:
		frames = f.getnframes()
		rate = f.getframerate()
		duration = frames / float(rate)
	return duration


def get_data_info(data_pool):
	"""
		data_pool: a list of path to LibriSpeech data subset
		return 
			data_info: { spk_0 : {
							wavs: [utt_0, utt_1], 
							durs : [dur_0, dur_1]},
						}
			transcript: { utt_0 : text_0, ...}
	"""
	data_info = {}
	transcript = {}
	for data_set in data_pool:
		for spk in os.listdir(data_set):
			data_info[spk] = {"wavs": [], "durs":[] }
			for book_chapter in os.listdir(os.path.join(data_set, spk)):
				for file in os.listdir(os.path.join(data_set, spk, book_chapter)):
					if(file.endswith(".wav")):
						wav_path = os.path.join(data_set, spk, book_chapter, file)
						data_info[spk]["wavs"].append(wav_path)
						data_info[spk]["durs"].append(get_duration(wav_path))
					if(file.endswith(".txt")):
						with open(os.path.join(data_set, spk, book_chapter, file)) as f:
							for line in f:
								tokens = line.strip().split(' ')
								transcript[tokens[0]] = ' '.join(tokens[1:])
	return data_info, transcript


def write_json(mixed_obj, outfile):
	outfile.write("{}\n".format(json.dumps(mixed_obj)))


def generate_jsonl(data_pool,
					number_utt=500000,
					spk_per_utt=2,
					set_name=None):
	if(set_name == None): 
		set_name = "{}mix".format(spk_per_utt)
	
	print('preparing metadata')
	global_spk_info = get_global_spk_info(os.path.join(data_root, "SPEAKERS.TXT"))
	data_info, transcripts = get_data_info(data_pool)
	spk_list = sorted(data_info.keys())

	print(f"writing data at list/{set_name}.jsonl")
	with open(f"list/{set_name}.jsonl", 'w') as outfile:
		for i in range(number_utt):
			wav_id = f"{set_name}/{set_name}_{str(i).zfill(len(str(number_utt)))}"
			picked_spk, genders = random_spk(spk_list, global_spk_info, n=2)
			wavs, durations = random_utterance(picked_spk, data_info)
			delays = random_delays(durations)
			texts = get_transcript(transcripts, wav_path=wavs)

			mixed_obj = {
				"id" : wav_id,
				"mixed_wav" : f"{wav_id}.wav",
				"speakers" : picked_spk,
				"texts" : texts,
				"wavs" : wavs,
				"delays" : delays,
				"durations": durations,
				"genders" : genders,
			}
			write_json(mixed_obj, outfile)




if __name__ == '__main__':
	number_utt = 20000
	spk_per_utt = 2
	data_root = "/home/burinn/dataset/librispeech/LibriSpeech"
	data_pool = ["train-clean-100", "train-clean-360", "train-other-500"]
	set_name = "train-{}mix".format(spk_per_utt)

	data_pool = [os.path.join(data_root, data_set) for data_set in data_pool]
	
	print(data_pool)
	generate_jsonl(data_pool, number_utt, spk_per_utt, set_name)


