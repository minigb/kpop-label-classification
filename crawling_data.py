import pandas as pd# This code is from "https://github.com/MALerLab/pop-era-classification/blob/main/crawling_data.py"
import yt_dlp
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from pathlib import Path
from argparse import ArgumentParser
import logging

from utils import get_song_id

# Configure logging
logging.basicConfig(filename='download_check.log', level=logging.INFO,
                    format='%(asctime)s:%(levelname)s:%(message)s')


class CsvColumnNames:
  def __init__(self, date, title, artist):
    self.date = date
    self.title = title
    self.artist = artist

class FileNames:
  def __init__(self, save_csv_name):
    self.queries = Path(f'{save_csv_name}_queries.csv')
    self.failed = Path(f'{save_csv_name}_failed.csv')
    self.chosen = Path(f'{save_csv_name}_chosen.csv')


class MusicCrawler:
  def __init__(self, input_csv_path: str, save_audio_dir: str, save_csv_name: str, exclude_keywords: list, include_keywords: list, query_suffix):
    self.input_csv_path = Path(input_csv_path)
    self.save_audio_dir = Path(save_audio_dir)
    self.save_csv_name = save_csv_name
    self.exclude_keywords = exclude_keywords
    self.include_keywords = include_keywords
    self.query_suffix = query_suffix
    self._custom_init()


  def _custom_init(self):
    self.in_csv_col_names = CsvColumnNames('Year', 'Song', 'Artist')
    self.out_csv_col_names = CsvColumnNames('Year', 'Song', 'Artist')

    self._init_save_csv_files()
    uniq_df = self._get_unique_df()
    self.target_df = self.get_df_with_existing_songs_removed(uniq_df)
    print(f"Number of songs to crawl: {len(self.target_df)}")


  def _init_save_csv_files(self):
    parent_dir = Path(self.save_csv_name).parent
    parent_dir.mkdir(parents=True, exist_ok=True)
    fns = FileNames(self.save_csv_name)

    # Check if files already exist
    if fns.queries.exists() or fns.failed.exists() or fns.chosen.exists():
      raise FileExistsError(f"One or more result files ({fns.queries}, {fns.failed}, {fns.chosen}) already exist. Please remove them before running the crawler.")
    
    # Create result files
    QUERIES_COLUMNS = [self.out_csv_col_names.date, self.out_csv_col_names.title, self.out_csv_col_names.artist] + ['query_idx', 'official', 'topic', 'channel_artist_same', 'keywords', 'video_title', 'video_channel', 'video_url']
    self.queries_df = pd.DataFrame(columns=QUERIES_COLUMNS)

    FAILED_COLUMNS = [self.out_csv_col_names.date, self.out_csv_col_names.title, self.out_csv_col_names.artist, 'Failed Reason']
    self.failed_df = pd.DataFrame(columns=FAILED_COLUMNS)

    CHOSEN_COLUMNS = [self.out_csv_col_names.date, self.out_csv_col_names.title, self.out_csv_col_names.artist] + ['query_idx', 'official', 'topic', 'channel_artist_same', 'keywords', 'video_title', 'video_channel', 'video_url']
    self.chosen_df = pd.DataFrame(columns=CHOSEN_COLUMNS)


  def _get_unique_df(self) -> pd.DataFrame:
    assert self.input_csv_path.exists(), f"{self.input_csv_path} does not exist"
    df = pd.read_csv(self.input_csv_path)

    # Sort by date
    df_sorted = df.sort_values(by=self.in_csv_col_names.date).reset_index(drop=True)
    df_sorted = df.sort_values(by=self.in_csv_col_names.date).reset_index(drop=True)

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
      video_title = video.get('title').lower()
      video_channel = video.get('channel') # In case of None, don't change into lower case
      video_duration = video.get('duration')
      video_url = video.get('url')
      
      if not (video_channel is not None and 60 <= video_duration <= 600):
        continue
      # Check if any exclude_keywords not in song_title are in video_title)
      if not (all(keyword.lower() not in video_title for keyword in exclude_keywords) \
                      and all(keyword.lower() in video_title for keyword in include_keywords)):
        continue
      if '/shorts/' in video_url:
        continue
      
      no_valid_video = False
      video_channel = video_channel.lower()
      video_info = {'query_idx': query_idx,
                    'channel_artist_same': artist.lower() in video_channel.replace(' - topic', '') or video_channel.replace(' - topic', '') in artist.lower(),
                    'video_title': video_title,
                    'video_channel': video_channel,
                    'video_url': video_url}
      queries.append(video_info)

    if no_valid_video:
      failed.append({'Failed Reason': 'Every channel is None or video duration is not between 60 and 600 seconds'})
        
    return song_ID, queries, failed


  def filter_queries_and_choose(self, queries, failed):
    NUM_OF_PRIORITIES = 4
    def _choose_video_with_priority(video, priority_num):
      def _is_official_audio():
        return 'official audio' in video['video_title']
      def _is_topic():
        return ' - topic' in video['video_channel']
      def _is_channel_artist_same():
        return video['channel_artist_same']
      
      assert 0 <= priority_num < NUM_OF_PRIORITIES
      is_satisfying = [
        _is_official_audio(),
        _is_channel_artist_same or _is_topic(),
        'color coded' in video['video_title'].replace('-', ' '),
        'lyric' in video['video_title'],
      ]
      assert len(is_satisfying) == NUM_OF_PRIORITIES, f"len(is_satisfying) should be {NUM_OF_PRIORITIES}, but got {len(is_satisfying)}"

      return is_satisfying[priority_num]

    chosen = None
    for i in range(NUM_OF_PRIORITIES):
      if chosen:
        break
      for video in queries:
        if _choose_video_with_priority(video, i):
          chosen = video
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
    queries_data = [{**{self.out_csv_col_names.date: date, self.out_csv_col_names.title: title, self.out_csv_col_names.artist: artist}, **video} for video in queries]
    self.queries_df = pd.concat([self.queries_df, pd.DataFrame(queries_data)], ignore_index=True)
    
    failed_data = [{**{self.out_csv_col_names.date: date, self.out_csv_col_names.title: title, self.out_csv_col_names.artist: artist}, **failure} for failure in failed]
    self.failed_df = pd.concat([self.failed_df, pd.DataFrame(failed_data)], ignore_index=True)
    
    if chosen:
      chosen_data = [{**{self.out_csv_col_names.date: date, self.out_csv_col_names.title: title, self.out_csv_col_names.artist: artist}, **chosen}]
      self.chosen_df = pd.concat([self.chosen_df, pd.DataFrame(chosen_data)], ignore_index=True)

    fns = FileNames(self.save_csv_name)
    for df, fn in zip([self.queries_df, self.failed_df, self.chosen_df], [fns.queries, fns.failed, fns.chosen]):
      df.to_csv(fn, index=False)


  def get_df_with_existing_songs_removed(self, df):
    print(f"Checking existing files for {self.save_csv_name}...")
    
    chosen_fn = FileNames(self.save_csv_name).chosen # check whether the result file exists
    refined_df_fn = f'{self.save_csv_name}_sorted_and_unique_chart.csv'
    if not chosen_fn.exists():
      # This means that the songs have not been crawled yet
      df.to_csv(refined_df_fn, index=False)
      return df
    
    chosen_df = pd.read_csv(chosen_fn)
    if chosen_df.empty:
      df.to_csv(refined_df_fn, index=False)
      return df

    # Remove existing songs to not crawl again
    refined_df = df
    existing_file_list = list(self.save_audio_dir.glob('*.mp3')) if self.save_audio_dir.exists() else []

    for _, row in tqdm(chosen_df.iterrows(), total=len(chosen_df)): # Get downloaded songs
      date = row[self.out_csv_col_names.date]
      title = row[self.out_csv_col_names.title]
      artist = row[self.out_csv_col_names.artist]

      matching_row = refined_df[(refined_df[self.in_csv_col_names.date] == date)
                                & (refined_df[self.in_csv_col_names.title] == title)
                                & (refined_df[self.in_csv_col_names.artist] == artist)]
      assert len(matching_row) == 1, f"len(matching_row) should be 1, but got {len(matching_row)}"
      song_id = get_song_id(matching_row[self.in_csv_col_names.date].values[0], matching_row[self.in_csv_col_names.title].values[0], matching_row[self.in_csv_col_names.artist].values[0])
      if f'{song_id}.mp3' in existing_file_list: # Remove only when the file exists
        refined_df = refined_df.drop(index=matching_row.index)

    for _, row in chosen_df.iterrows():
      date = row[self.out_csv_col_names.date]
      title = row[self.out_csv_col_names.title]
      artist = row[self.out_csv_col_names.artist]
      song_id = get_song_id(date, title, artist)
      fn = f'{song_id}.mp3'
      assert fn in existing_file_list

    refined_df.to_csv(refined_df_fn, index=False)
    return refined_df
            
            
  def run(self, topk):
    song_list = [(row[self.in_csv_col_names.title], row[self.in_csv_col_names.artist], row[self.in_csv_col_names.date], self.exclude_keywords, topk, self.include_keywords) \
                 for _, row in self.target_df.iterrows()]

    self.run_parallel(song_list) # Results are saved while crawling

    # Check if all chosen songs have corresponding mp3 files
    self.check_downloaded_files()

  def check_downloaded_files(self):
    chosen_fn = FileNames(self.save_csv_name).chosen
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
  

class FailedMusicCrawler(MusicCrawler):
  def __init__(self, input_csv_path, save_audio_dir, save_csv_name, exclude_keywords, include_keywords, query_suffix):
    super().__init__(input_csv_path, save_audio_dir, save_csv_name, exclude_keywords, include_keywords, query_suffix)


  def _custom_init(self):
    for fn in FileNames(self.save_csv_name).__dict__.values():
      assert fn.exists(), f"{fn} does not exist"
    self.input_csv_path = FileNames(self.save_csv_name).failed

    self.in_csv_col_names = self.out_csv_col_names = CsvColumnNames('Year', 'Song', 'Artist')
    self._init_save_csv_files()

    self.target_df = self._get_unique_df()  # use self.input_csv_path
    print(f"Number of failed songs to re-crawl: {len(self.target_df)}")


class ReusingQueriesMusicCrawler(MusicCrawler):
  def __init__(self, input_csv_path, save_audio_dir, save_csv_name, exclude_keywords, include_keywords):
    super().__init__(input_csv_path, save_audio_dir, save_csv_name, exclude_keywords, include_keywords, None)


  def _custom_init(self):
    for fn in FileNames(self.save_csv_name).__dict__.values():
      assert fn.exists(), f"{fn} does not exist"
    self.input_csv_path = FileNames(self.save_csv_name).failed
    self.in_csv_col_names = self.out_csv_col_names = CsvColumnNames('Year', 'Song', 'Artist')

    self._init_save_csv_files()
    self.target_df = self.get_df_with_existing_songs_removed()
    print(f"Number of songs to crawl: {len(self.target_df)}")


  def make_query(self, song, artist, date, _1, _2, _3):
    song_ID = self._get_song_identifier(date, song, artist)

    queries_fn = FileNames(self.save_csv_name).queries
    queries_df = pd.read_csv(queries_fn)
    queries_df = queries_df[(queries_df[self.out_csv_col_names.date] == date) \
                            & (queries_df[self.out_csv_col_names.title] == song) \
                            & (queries_df[self.out_csv_col_names.artist] == artist)]
    queries_df = queries_df.drop(columns = [self.out_csv_col_names.date, self.out_csv_col_names.title, self.out_csv_col_names.artist])
    queries = queries_df.to_dict(orient='records')

    return song_ID, queries, []


if __name__ == '__main__':
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

  EXCLUDE_KEYWORDS_REMASTER_NOT_INCLUDED = ['live', 'cover', 'mv', 'm/v', 'video', 'mix', 'ver', 'version', 'remix', 'remake', 'arrange', 'dj', 'practice', 'inst', 'teaser', 'performance', 'karaoke', 'inkigayo', '음악중심']
  REMASTER = 'remaster'

  argparser = ArgumentParser()
  argparser.add_argument('--input_csv', type=str)
  argparser.add_argument('--save_csv_name', type=str, required=True)
  argparser.add_argument('--save_audio_dir', type=str, required=True)
  argparser.add_argument('--exclude_remaster', action='store_true')
  argparser.add_argument('--include_remaster', action='store_true') # TODO: Find better approach
  argparser.add_argument('--topk', type=int, default=10)
  argparser.add_argument('--crawler_type', type=str, default='new')
  argparser.add_argument('--query_suffix', type=str)
  args = argparser.parse_args()

  exclude_keywords = EXCLUDE_KEYWORDS_REMASTER_NOT_INCLUDED
  if args.exclude_remaster:
    exclude_keywords.append(REMASTER)
  
  include_keywords = []
  if args.include_remaster:
    include_keywords = [REMASTER]

  # Select crawler
  if args.crawler_type == 'new':
    fns = FileNames(args.save_csv_name)
    assert not fns.chosen.exists(), f"{fns.chosen} already exists. If you are trying to recrawl, use 'failed' as the crawler type."
    crawler = MusicCrawler(args.input_csv, args.save_audio_dir, args.save_csv_name, exclude_keywords, include_keywords, args.query_suffix)
  elif args.crawler_type == 'failed':
    crawler = FailedMusicCrawler(args.save_audio_dir, args.save_csv_name, exclude_keywords, include_keywords, args.query_suffix)
  elif args.crawler_type == 'reuse':
    crawler = ReusingQueriesMusicCrawler(args.save_audio_dir, args.save_csv_name, exclude_keywords, include_keywords)
  else:
    raise ValueError(f"Invalid crawler type: {args.crawler_type}")
  
  # Run
  crawler.run(args.topk)