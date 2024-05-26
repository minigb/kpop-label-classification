# This code is from "https://github.com/MALerLab/pop-era-classification/blob/main/crawling_data.py"

import pandas as pd
import yt_dlp
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
from pathlib import Path
from argparse import ArgumentParser

from utils import get_song_id


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


class BillboardMusicCrawler:
  def __init__(self, input_csv_path: str, save_audio_dir: str, save_csv_name: str, exclude_keywords: list, include_keywords: list, query_suffix):
    self.input_csv_path = Path(input_csv_path)
    self.save_audio_dir = Path(save_audio_dir)
    self.save_csv_name = save_csv_name
    self.exclude_keywords = exclude_keywords
    self.include_keywords = include_keywords

    self.in_csv_col_names = CsvColumnNames('Year', 'Song', 'Artist')
    self.out_csv_col_names = CsvColumnNames('Year', 'Song', 'Artist')

    self._init_result_csv()
    self.input_df = self.get_sorted_and_unique_billboard_df()
    self.target_df = self.get_df_with_existing_songs_removed(self.input_df)

    self.query_suffix = query_suffix


  def _init_result_csv(self):
    parent_dir = Path(self.save_csv_name).parent
    parent_dir.mkdir(parents=True, exist_ok=True)
    fns = FileNames(self.save_csv_name)

    # queries result df
    QUERIES_COLUMNS = [self.out_csv_col_names.date, self.out_csv_col_names.title, self.out_csv_col_names.artist] + ['query_idx', 'official', 'topic', 'channel_artist_same', 'keywords', 'video_title', 'video_channel', 'video_url']
    if fns.queries.exists():
      queries_df = pd.read_csv(fns.queries)
      assert set(queries_df.columns) == set(QUERIES_COLUMNS), f"columns of {fns.queries} should be {QUERIES_COLUMNS}, but got {queries_df.columns}"
    else:
      queries_df = pd.DataFrame(columns=QUERIES_COLUMNS)
    self.queries_df = queries_df

    # failed result df
    FAILED_COLUMNS = [self.out_csv_col_names.date, self.out_csv_col_names.title, self.out_csv_col_names.artist, 'Failed Reason']
    self.failed_df = pd.DataFrame(columns=FAILED_COLUMNS)

    # chosen result df
    CHOSEN_COLUMNS = [self.out_csv_col_names.date, self.out_csv_col_names.title, self.out_csv_col_names.artist] + ['query_idx', 'official', 'topic', 'channel_artist_same', 'keywords', 'video_title', 'video_channel', 'video_url']
    if fns.chosen.exists():
      chosen_df = pd.read_csv(fns.chosen)
      assert set(chosen_df.columns) == set(CHOSEN_COLUMNS)
    else:
      chosen_df = pd.DataFrame(columns=CHOSEN_COLUMNS)
    self.chosen_df = chosen_df

    
  def get_sorted_and_unique_billboard_df(self) -> pd.DataFrame:
    assert self.input_csv_path.exists(), f"{self.input_csv_path} does not exist"
    df = pd.read_csv(self.input_csv_path)

    # Sort by date
    df_sorted = df.sort_values(by=self.in_csv_col_names.date).reset_index(drop=True)

    # Remove duplicates
    df_uniq = df_sorted.drop_duplicates(subset=[self.in_csv_col_names.title, self.in_csv_col_names.artist], keep='first')
    
    return df_uniq
  

  def make_query(self, song, artist, date, exclude_keywords, topk, include_keywords):
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
      song_ID = self._get_song_identifier(date, song, artist)
      
      if video_channel is not None and 60 <= video_duration <= 600:
        no_valid_video = False
        video_channel = video_channel.lower()
        keywords = all(keyword.lower() not in video_title for keyword in exclude_keywords) \
                       and all(keyword.lower() in video_title for keyword in include_keywords) # Check if any exclude_keywords not in song_title are in video_title
        video_info = {'query_idx': query_idx,
                      'official': 'official audio' in video_title,
                      'topic': ' - topic' in video_channel,
                      'channel_artist_same': artist.lower() in video_channel.replace(' - topic', '') or video_channel.replace(' - topic', '') in artist.lower(),
                      'keywords': keywords,
                      'video_title': video_title,
                      'video_channel': video_channel,
                      'video_url': video.get('url')}
        queries.append(video_info)

    if no_valid_video:
      failed.append({'Failed Reason': 'Every channel is None or video duration is not between 60 and 600 seconds'})
        
    return song_ID, queries, failed


  def filter_queries_and_choose(self, queries, failed):
    PRIORITIES = 4
    def _choose_video_with_priority(video, priority_num):
      assert 0 <= priority_num < PRIORITIES
      is_satisfying = [
        video['official'] and video['keywords'],
        'color coded lyrics' in video['title'],
        (video['channel_artist_same'] and video['keywords']) or (video['topic'] and video['keywords']),
        video['channel_artist_same'] and video['topic'] and video['keywords']
      ]
      assert len(is_satisfying) == PRIORITIES, f"{len(is_satisfying)} != {PRIORITIES}"

      return is_satisfying[priority_num]

    chosen = None
    for i in range(PRIORITIES):
      for video in queries:
        if _choose_video_with_priority(video, i):
          chosen = video
          break
    # # 'official audio' in video_title in top 3 queries
    # if any([video for video in queries[:3] if video['official'] and video['keywords']]):
    #   chosen = [video for video in queries[:3] if video['official'] and video['keywords']][0]
    # # video channel and artist name are the same or video channel is topic channel in top 3 queries
    # elif any([video for video in queries[:3] if (video['channel_artist_same'] and video['keywords']) or (video['topic'] and video['keywords'])]):
    #   chosen = [video for video in queries[:3] if (video['channel_artist_same'] and video['keywords']) or (video['topic'] and video['keywords'])][0]
    # # video channel and artist name are the same AND video channel is topic channel in top 4~10 queries
    # elif any([video for video in queries[3:] if video['channel_artist_same'] and video['topic'] and video['keywords']]):
    #   chosen = [video for video in queries[3:] if video['channel_artist_same'] and video['topic'] and video['keywords']][0]
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
      'outtmpl': f'{self.save_audio_dir}/{song_id}',
      'postprocessors': [{
        'key': 'FFmpegExtractAudio',
        'preferredcodec': 'mp3',
        'preferredquality': '192'
      }]
    }
    with yt_dlp.YoutubeDL(download_opts) as ydl:
      ydl.download([chosen['video_url']])
  
  
  def process_song(self, song):
    song_ID, queries, failed = self.make_query(*song)
    chosen, failed_update = self.filter_queries_and_choose(queries, failed)
    if chosen:
      try:
        self.download_video(chosen, song_ID)
      except Exception as e:
        print(e)
    return song_ID, queries, failed_update, chosen


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
    print(f"Number of songs to crawl: {len(song_list)}")

    self.run_parallel(song_list) # Results are saved while crawling

  def _get_song_identifier(self, date, song, artist):
    # This is different from song_id used for file name
    return f'{date}@@{song}@@{artist}'
  
  def _decode_song_identifier(self, song_ID):
    date, song, artist = song_ID.split('@@')
    return date, song, artist
  

class FailedMusicCrawler(BillboardMusicCrawler):
  def __init__(self, save_audio_dir, save_csv_name, exclude_keywords, include_keywords, query_suffix):
    self.save_audio_dir = save_audio_dir
    self.save_csv_name = save_csv_name
    self.exclude_keywords = exclude_keywords
    self.include_keywords = include_keywords
    self.query_suffix = query_suffix
    
    for fn in FileNames(self.save_csv_name).__dict__.values():
      assert fn.exists(), f"{fn} does not exist"
    self.input_csv_path = FileNames(self.save_csv_name).failed
    self.in_csv_col_names = self.out_csv_col_names = CsvColumnNames('Year', 'Song', 'Artist')

    self._init_result_csv()
    self.target_df = self.get_sorted_and_unique_billboard_df()


# TODO(minigb): Do not use RecrawlerForRemastered. This is not guaranteed to work.
class RecrawlerForRemastered(BillboardMusicCrawler):
  REMASTER = 'remaster'

  def __init__(self, input_csv_path: str, save_audio_dir: str, save_csv_name: str, exclude_keywords: list):
    exclude_keywords = exclude_keywords + [self.REMASTER] if self.REMASTER not in exclude_keywords else exclude_keywords

    self.input_csv_path = Path(input_csv_path)
    self.save_audio_dir = Path(save_audio_dir)
    self.save_csv_name = save_csv_name
    self.exclude_keywords = exclude_keywords

    self.in_csv_col_names = CsvColumnNames('Date', 'Song', 'Artist')
    self.out_csv_col_names = CsvColumnNames('Date', 'Song', 'Artist')

    self._init_result_csv()
    self.input_df = pd.read_csv(self.input_csv_path)
    self.target_df = self._leave_only_remastered_song(self.input_df)


  def _leave_only_remastered_song(self, df):
    indexs_to_drop = []
    for index, row in df.iterrows():
      video_title = row['video_title']
      song, artist = row[self.out_csv_col_names.title], row[self.out_csv_col_names.artist]
      if REMASTER in video_title.lower():
        pass
      if not (REMASTER in video_title.lower() and not REMASTER in song.lower() and not REMASTER in artist.lower()):
        indexs_to_drop.append(index)

    df = df.drop(indexs_to_drop)
    return df

    
if __name__ == '__main__':
  """
  Example line to run the code: 
  python crawling_data.py --input_csv song_list.csv --save_csv_name csv/kpop --save_audio_dir audio

  When recrawling failed music:
  python crawling_data.py --save_csv_name csv/kpop --save_audio_dir audio --crawler_type failed --query_suffix "official audio"

  When recrawling remastered music:
  python crawling_data.py --input_csv csv/kpop_chosen.csv --save_csv_name csv/kpop_remaster --save_audio_dir audio_remastered_original --crawler_type remastered
  
  Add '--exclude_remaster' or '--topk 20' if needed
  """

  EXCLUDE_KEYWORDS_REMASTER_NOT_INCLUDED = ['live', 'cover', 'mv', 'video', 'mix', 'version', 'remix', 'remake', 'arrange', 'dj']
  REMASTER = 'remaster'

  argparser = ArgumentParser()
  argparser.add_argument('--input_csv', type=str)
  argparser.add_argument('--save_csv_name', type=str, required=True)
  argparser.add_argument('--save_audio_dir', type=str, required=True)
  argparser.add_argument('--exclude_remaster', action='store_true')
  argparser.add_argument('--include_remaster', action='store_true') # TODO(minigb): Find better approach
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
    crawler = BillboardMusicCrawler(args.input_csv, args.save_audio_dir, args.save_csv_name, exclude_keywords, include_keywords)
  elif args.crawler_type == 'failed':
    crawler = FailedMusicCrawler(args.save_audio_dir, args.save_csv_name, exclude_keywords, include_keywords, args.query_suffix)
  elif args.crawler_type == 'testset':
    crawler = FailedMusicCrawler(args.save_audio_dir, args.save_csv_name, exclude_keywords, include_keywords, args.query_suffix)
  # elif args.crawler_type == 'remastered': # TODO(minigb): Do not use RecrawlerForRemastered. This is not guaranteed to work.
  #   crawler = RecrawlerForRemastered(args.input_csv, args.save_audio_dir, args.save_csv_name, exclude_keywords)
  else:
    raise ValueError(f"Invalid crawler type: {args.crawler_type}")
  
  # Run
  crawler.run(args.topk)