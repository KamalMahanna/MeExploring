from bs4 import BeautifulSoup
import asyncio
from multiprocessing import Process, current_process
from aiohttp import ClientSession, ClientTimeout
import pandas as pd
import time

df = pd.read_csv('all_anime_urls.csv')


async def fetch_urls(url,session,a_list_storage,failed_ones):
    
    try:
        async with session.get(url, timeout=ClientTimeout(total=10)) as response:
            if response.status != 200:
                raise Exception(response.status)
            html_response = await response.text()
            soup =  BeautifulSoup(html_response, 'html.parser')
            
            others = soup.find_all('div',attrs={'class':'spaceit_pad'})
            others = [' '.join(i.text.strip().split()) for i in others]
            
            others_dict = {i.split(':')[0].strip():' '.join(i.split(':')[1:]).strip() 
                           for i in others if len(i.split(':')) > 1}
            
            Synopsis = soup.find('p',attrs={'itemprop':'description'}).text
            others_dict['Synopsis'] = Synopsis if Synopsis else None
            others_dict['url'] = url
            a_list_storage.append(others_dict)
            
    except Exception as e:
        print(e)
        failed_ones.update([url])    
        
async def process_batch(urls, a_list_storage,failed_ones):
    async with ClientSession() as session:
        tasks = [fetch_urls(url,
                            session,a_list_storage,failed_ones) 
                 for url in urls]
        return await asyncio.gather(*tasks)
    
failed_ones = set()
import pandas as pd
# df = pd.read_csv('all_anime_urls.csv',chunksize=50)
df = pd.read_csv('all_failed_urls.csv',chunksize=50)

# first = True
first = False

for each_df in df:
    all_anime_details = []
    asyncio.run(process_batch(each_df['url'].to_list(),all_anime_details,failed_ones))
    anime_details_df = pd.DataFrame(all_anime_details)
    anime_details_df.to_csv('anime_details.csv', mode='a', header=first, index=False)
    first = False
    # time.sleep(10)
    
df_failed = pd.DataFrame(failed_ones,columns=['url'])
df_failed.to_csv('all_failed_urls.csv',index=False)
print('ok')


