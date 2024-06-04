# This code an updated vesion of "https://github.com/MALerLab/pop-era-classification/blob/main/crawling_data.py"

import pandas as pd
import yt_dlp
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from pathlib import Path
import logging
import json
import hydra

from utils import get_song_id
from dataset_cleanup import AudioDownloadCleaner

# TODO(minigb): This is not ideal. Find a better way to handle this.
VIDEO_INFO_COLUMNS = ['query_idx', 'channel_artist_same', 'video_title', 'video_channel', 'video_url']

# Configure logging
logging.basicConfig(filename='download_check.log', level=logging.INFO,
                    format='%(asctime)s:%(levelname)s:%(message)s')


class MusicCrawler:
  def __init__(self, config, exclude_keywords, include_keywords, query_suffix):
    self.config = config
    self.input_csv_path = Path(config.kpop_dataset.song_list_csv_fn)
    self.save_audio_dir = Path(config.data.audio_dir)
    self.save_csv_fns = config.kpop_dataset.save_csv_fns
    self.exclude_keywords = exclude_keywords
    self.include_keywords = include_keywords
    self.query_suffix = query_suffix
    self.csv_column_names = config.csv_column_names
    self._custom_init()


  def _custom_init(self):
    assert not Path(self.save_csv_fns.chosen).exists(), f"{self.save_csv_fns.chosen} already exists. If you are trying to recrawl, use 'failed' as the crawler type."

    self.in_csv_col_names = self.csv_column_names.song
    self.out_csv_col_names = self.csv_column_names.video

    self._init_save_csv_files()
    self.target_df = self._remove_existing_songs_from_the_input_df()
    print(f"Number of songs to crawl: {len(self.target_df)}")


  def _init_save_csv_files(self):
    # Create parent directory of save_csv_fns    
    Path(self.save_csv_fns.chosen).parent.mkdir(parents=True, exist_ok=True)

    # queries result df
    # if the file exists, read it and check if the columns are correct
    # if not, create an empty dataframe with the correct columns
    QUERIES_COLUMNS = [self.out_csv_col_names.year, self.out_csv_col_names.title, self.out_csv_col_names.artist] + VIDEO_INFO_COLUMNS
    if Path(self.save_csv_fns.queries).exists():
      queries_df = pd.read_csv(self.save_csv_fns.queries)
      assert set(queries_df.columns) == set(QUERIES_COLUMNS)
    else:
      queries_df = pd.DataFrame(columns=QUERIES_COLUMNS)
    self.queries_df = queries_df

    # failed result df
    # create an empty dataframe with the correct columns
    FAILED_COLUMNS = [self.out_csv_col_names.year, self.out_csv_col_names.title, self.out_csv_col_names.artist, 'Failed Reason']
    self.failed_df = pd.DataFrame(columns=FAILED_COLUMNS)

    # chosen result df
    # if the file exists, read it and check if the columns are correct
    # if not, create an empty dataframe with the correct columns
    CHOSEN_COLUMNS = [self.out_csv_col_names.year, self.out_csv_col_names.title, self.out_csv_col_names.artist] + VIDEO_INFO_COLUMNS
    if Path(self.save_csv_fns.chosen).exists():
      chosen_df = pd.read_csv(self.save_csv_fns.chosen)
      assert set(chosen_df.columns) == set(CHOSEN_COLUMNS)
    else:
      chosen_df = pd.DataFrame(columns=CHOSEN_COLUMNS)
    self.chosen_df = self._remove_rows_without_audio(chosen_df)


  def _remove_rows_without_audio(self, df):
    idx_to_remove = []
    for idx, row in df.iterrows():
      date = row[self.out_csv_col_names.year]
      title = row[self.out_csv_col_names.title]
      artist = row[self.out_csv_col_names.artist]
      song_id = get_song_id(date, title, artist)
      if not Path(f"{self.save_audio_dir}/{song_id}.mp3").exists():
        idx_to_remove.append(idx)
    return df.drop(index=idx_to_remove)


  def _get_df_with_unique_songs(self) -> pd.DataFrame:
    assert Path(self.input_csv_path).exists(), f"{self.input_csv_path} does not exist"
    df = pd.read_csv(self.input_csv_path)

    # Sort by date
    df_sorted = df.sort_values(by=self.in_csv_col_names.year).reset_index(drop=True)

    # Remove duplicates
    df_uniq = df_sorted.drop_duplicates(subset=[self.in_csv_col_names.title, self.in_csv_col_names.artist], keep='first')
    
    return df_uniq
  

  def make_query(self, song, artist, date, exclude_keywords, topk, include_keywords):
    song_ID = self._get_song_identifier(date, song, artist)

    ydl_opts = {
      'quiet': True,
      'skip_download': True,
      'extract_flat': True,
    }
    query = f"{artist} {song} {self.query_suffix}" if self.query_suffix else f"{artist} {song}"
    queries, failed = [], []
    
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
      info = ydl.extract_info(f"ytsearch{topk}:{query}", download=False)
    
    no_valid_video = True
    for query_idx, video in enumerate(info['entries']):
      # Add video info to queries
      video_title = video.get('title').lower()
      video_channel = video.get('channel') # In case of None, don't change into lower case
      video_duration = video.get('duration')
      video_url = video.get('url')
      video_channel = video_channel.lower()
      video_info = {'query_idx': query_idx,
                    'channel_artist_same': artist.lower() in video_channel.replace(' - topic', '') or video_channel.replace(' - topic', '') in artist.lower(),
                    'video_title': video_title,
                    'video_channel': video_channel,
                    'video_url': video_url}
      queries.append(video_info)
      
      # Check if video is valid
      if not (video_channel is not None and 60 <= video_duration <= 600):
        continue
      # Check if any exclude_keywords not in song_title are in video_title)
      if not (all(keyword.lower() not in video_title for keyword in exclude_keywords) \
                      and all(keyword.lower() in video_title for keyword in include_keywords)):
        continue
      if '/shorts/' in video_url:
        continue
      
      no_valid_video = False

    if no_valid_video:
      failed.append({'Failed Reason': 'Every channel is None or video duration is not between 60 and 600 seconds'})
        
    return song_ID, queries, failed


  def filter_queries_and_choose(self, queries, failed):
    def _is_official_audio():
      return 'official audio' in video['video_title']
    def _is_topic():
      return ' - topic' in video['video_channel']
    def _is_channel_artist_same():
      return video['channel_artist_same']
    def _is_lyric_video():
      for keyword in ['color coded', 'lyric', '가사']:
        if keyword in video['video_title'].replace('-', ' '):
          return True
    
    is_satisfying = [
      _is_official_audio(),
      _is_channel_artist_same or _is_topic(),
      _is_lyric_video(),
    ]

    chosen = None
    for check_func in is_satisfying:
      for video in queries:
        if check_func(video):
          chosen = video
          break
      if chosen:
        break

    if not chosen:
      video_info = {'Failed Reason': 'No suitable video'}
      failed.append(video_info)

    return chosen, failed


  def download_video(self, chosen, song_ID):
    print(f"Downloading {chosen['video_title']}...")
    date, song, artist = self._decode_song_identifier(song_ID)
    song_id = get_song_id(date, song, artist)
    download_opts = {
      'format': 'bestaudio/best',
      'outtmpl': f'{self.save_audio_dir}/{song_id}.%(ext)s',
      'postprocessors': [{
        'key': 'FFmpegExtractAudio',
        'preferredcodec': 'mp3',
        'preferredquality': '192'
      }]
    }
    with yt_dlp.YoutubeDL(download_opts) as ydl:
      ydl.download([chosen['video_url']])
      # Verify the file was downloaded
      if not Path(f"{self.save_audio_dir}/{song_id}.mp3").exists():
          raise Exception(f"Failed to download audio for {song_ID}")


  def process_song(self, song):
    song_ID, queries, failed = self.make_query(*song)
    chosen, failed_update = self.filter_queries_and_choose(queries, failed)
    if chosen:
      try:
        self.download_video(chosen, song_ID)
        # If download is successful, return chosen
        return song_ID, queries, failed_update, chosen
      except Exception as e:
        print(e)
        failed_update.append({'Failed Reason': str(e)})
    # Return None for chosen if download fails or no video is chosen
    return song_ID, queries, failed_update, None


  def run_parallel(self, song_list):
    with ThreadPoolExecutor(max_workers=16) as executor:
      future_to_song = {executor.submit(self.process_song, song): song for song in song_list}
      for future in tqdm(as_completed(future_to_song), total=len(song_list)):
        song = future_to_song[future]
        try:
          song_ID, queries, failed, chosen = future.result()
          self.update_result_csv(song_ID, queries, failed, chosen)
        except Exception as exc:
          print(f'{song} generated an exception: {exc}')


  def update_result_csv(self, song_ID, queries, failed, chosen):
    date, title, artist = self._decode_song_identifier(song_ID)

    # queries dict
    queries_data = [{**{self.out_csv_col_names.year: date, self.out_csv_col_names.title: title, self.out_csv_col_names.artist: artist}, **video} for video in queries]
    self.queries_df = pd.concat([self.queries_df, pd.DataFrame(queries_data)], ignore_index=True)
    
    failed_data = [{**{self.out_csv_col_names.year: date, self.out_csv_col_names.title: title, self.out_csv_col_names.artist: artist}, **failure} for failure in failed]
    self.failed_df = pd.concat([self.failed_df, pd.DataFrame(failed_data)], ignore_index=True)
    
    if chosen:
      chosen_data = [{**{self.out_csv_col_names.year: date, self.out_csv_col_names.title: title, self.out_csv_col_names.artist: artist}, **chosen}]
      self.chosen_df = pd.concat([self.chosen_df, pd.DataFrame(chosen_data)], ignore_index=True)

    for df, fn in zip([self.queries_df, self.failed_df, self.chosen_df], [self.save_csv_fns.queries, self.save_csv_fns.failed, self.save_csv_fns.chosen]):
      df.to_csv(fn, index=False)


  def _remove_existing_songs_from_the_input_df(self):
    input_df = self._get_df_with_unique_songs()

    if self.chosen_df.empty:
      return input_df

    # Remove existing songs to not crawl again
    # self.chosen_df is already filtered to remove songs without audio

    idx_to_remove = []
    for _, row in tqdm(self.chosen_df.iterrows(), total=len(self.chosen_df)): # Get downloaded songs
      date = row[self.out_csv_col_names.year]
      title = row[self.out_csv_col_names.title]
      artist = row[self.out_csv_col_names.artist]

      matching_row = input_df[(input_df[self.in_csv_col_names.year] == date)
                                & (input_df[self.in_csv_col_names.title] == title)
                                & (input_df[self.in_csv_col_names.artist] == artist)]
      assert len(matching_row) == 1, f"len(matching_row) should be 1, but got {len(matching_row)}"
      idx_to_remove.append(matching_row.index[0])

    input_df = input_df.drop(index=idx_to_remove)

    return input_df
            
            
  def run(self, topk):
    song_list = [(row[self.in_csv_col_names.title], row[self.in_csv_col_names.artist], row[self.in_csv_col_names.year], self.exclude_keywords, topk, self.include_keywords) \
                 for _, row in self.target_df.iterrows()]

    self.run_parallel(song_list) # Results are saved while crawling

    # Check if all chosen songs have corresponding mp3 files
    self.check_downloaded_files()

  def check_downloaded_files(self):
    chosen_fn = self.save_csv_fns.chosen
    if not chosen_fn.exists():
        logging.info("No chosen file found.")
        return

    chosen_df = pd.read_csv(chosen_fn)
    total_chosen = len(chosen_df)
    actual_files = sum(1 for _ in self.save_audio_dir.glob('*.mp3'))

    logging.info(f"Number of songs noted as succeeded: {total_chosen}")
    logging.info(f"Number of actual MP3 files: {actual_files}")

  def _get_song_identifier(self, date, song, artist):
    # This is different from the song_id used for file name
    return f'{date}@@{song}@@{artist}'
  
  def _decode_song_identifier(self, song_ID):
    date, song, artist = song_ID.split('@@')
    return date, song, artist
  

class AdditionalMusicCrawler(MusicCrawler):
  def __init__(self, config, exclude_keywords, include_keywords, query_suffix):
    super().__init__(config, exclude_keywords, include_keywords, query_suffix)


  def _custom_init(self):
    for fn in [self.save_csv_fns.chosen, self.save_csv_fns.queries, self.save_csv_fns.failed]: # TODO(minigb): Better way to handle this?
      assert Path(fn).exists(), f"{fn} does not exist"

    # Remove noisy results
    cleaner = AudioDownloadCleaner(self.config)
    cleaner.run()

    self.in_csv_col_names = self.csv_column_names.song
    self.out_csv_col_names = self.csv_column_names.video
 
    self._init_save_csv_files()
    self.target_df = self._remove_existing_songs_from_the_input_df()
    print(f"Number of failed songs to additionally crawl: {len(self.target_df)}")


# TODO(minigb): Implement this
# class ReusingQueriesMusicCrawler(MusicCrawler):
#   def __init__(self, input_csv_path, save_audio_dir, save_csv_fns, exclude_keywords, include_keywords):
#     super().__init__(input_csv_path, save_audio_dir, save_csv_fns, exclude_keywords, include_keywords, None)


#   def _custom_init(self):
#     for fn in self.save_csv_fns.__dict__.values():
#       assert fn.exists(), f"{fn} does not exist"
#     self.input_csv_path = self.save_csv_fns.failed
#     self.in_csv_col_names = self.out_csv_col_names = self.csv_column_names.video

#     self._init_save_csv_files()
#     self.target_df = self._remove_existing_songs_from_the_input_df()
#     print(f"Number of songs to crawl: {len(self.target_df)}")


#   def make_query(self, song, artist, date, _1, _2, _3):
#     song_ID = self._get_song_identifier(date, song, artist)

#     queries_fn = self.save_csv_fns.queries
#     queries_df = pd.read_csv(queries_fn)
#     queries_df = queries_df[(queries_df[self.out_csv_col_names.year] == date) \
#                             & (queries_df[self.out_csv_col_names.title] == song) \
#                             & (queries_df[self.out_csv_col_names.artist] == artist)]
#     queries_df = queries_df.drop(columns = [self.out_csv_col_names.year, self.out_csv_col_names.title, self.out_csv_col_names.artist])
#     queries = queries_df.to_dict(orient='records')

#     return song_ID, queries, []


def update_config(config):
  if not config.input_csv:
    config.input_csv = config.kpop_dataset.song_list_csv_fn
  if config.save_csv_name:
    # TODO(minigb): Better way to handle this?
    config.kpop_dataset.save_csv_fns.chosen = f"{config.save_csv_name}_chosen.csv"
    config.kpop_dataset.save_csv_fns.queries = f"{config.save_csv_name}_queries.csv"
    config.kpop_dataset.save_csv_fns.failed = f"{config.save_csv_name}_failed.csv"
  if config.save_audio_dir:
    config.data.audio_dir = config.save_audio_dir


@hydra.main(config_path='config', config_name='packed')
def main(config):
  """
  Example line to run the code: 
  python crawling_data.py --input_csv kpop-dataset/song_list.csv --save_csv_name kpop-dataset/csv/kpop --save_audio_dir audio

  When recrawling failed music, using query suffix:
  python crawling_data.py --save_csv_name csv/kpop --save_audio_dir audio --crawler_type failed --query_suffix "official audio"

  When recrawling failed music with reuse of queries:
  python crawling_data.py --save_csv_name csv/kpop --save_audio_dir audio --crawler_type reuse

  When recrawling remastered music:
  python crawling_data.py --input_csv csv/kpop_chosen.csv --save_csv_name csv/kpop_remaster --save_audio_dir audio_remastered_original --crawler_type remastered
  python crawling_data.py --input_csv csv/kpop_chosen.csv --save_csv_name csv/kpop_remaster --save_audio_dir audio_remastered_original --crawler_type remastered
  
  Add '--exclude_remaster' or '--topk 20' if needed
  """
  REMASTER = 'remaster'

  update_config(config)

  exclude_keywords_path = Path(config.exclude_keywords.video_title_fn)
  with exclude_keywords_path.open('r') as f:
    exclude_keywords = json.load(f)
  
  if config.exclude_remaster:
    exclude_keywords.append(REMASTER)
  
  include_keywords = []
  if config.include_remaster:
    include_keywords = [REMASTER]

  # Select crawler
  if config.crawler_type == 'new':
    crawler = MusicCrawler(config, exclude_keywords, include_keywords, config.query_suffix)
  elif config.crawler_type == 'additional':
    crawler = AdditionalMusicCrawler(config, exclude_keywords, include_keywords, config.query_suffix)
  # elif config.crawler_type == 'reuse':
  #   crawler = ReusingQueriesMusicCrawler(config, exclude_keywords, include_keywords, config.query_suffix)
  else:
    raise ValueError(f"Invalid crawler type: {config.crawler_type}")
  
  # Run
  crawler.run(config.topk)
  
if __name__ == '__main__':
  main()