from bs4 import BeautifulSoup
import asyncio
from multiprocessing import Process, current_process
from aiohttp import ClientSession, ClientTimeout

async def fetch_urls(url,session,a_set_storage,failed_ones):
    
    try:
        async with session.get(url, timeout=ClientTimeout(total=10)) as response:
            if response.status != 200:
                raise Exception(response.status,"Something went wrong")
            html_response = await response.text()
            soup =  BeautifulSoup(html_response, 'html.parser')
            a_set_storage.update(i.get('href') for i in soup.find_all('a',attrs={'class':'hoverinfo_trigger'}))
    except Exception as e:
        print(e)
        failed_ones.update([url])    
        
async def process_batch(start,end, a_set_storage,failed_ones):
    async with ClientSession() as session:
        tasks = [fetch_urls(f'https://myanimelist.net/topanime.php?type=bypopularity&limit={no}',
                            session,a_set_storage,failed_ones) 
                 for no in range(start,end,50)]
        return await asyncio.gather(*tasks)
    
all_anime_urls = set()
failed_ones = set()

for i in zip(range(0,28701,2500),range(2500,28701,2500)):
    print(i,len(failed_ones))
    asyncio.run(process_batch(i[0],i[1],all_anime_urls,failed_ones))
    
asyncio.run(process_batch(27500,28701,all_anime_urls,failed_ones))

import pandas as pd

df = pd.DataFrame(all_anime_urls,columns=['url'])
df.to_csv('all_anime_urls.csv',index=False)
df_failed = pd.DataFrame(failed_ones,columns=['url'])
df_failed.to_csv('all_failed_urls.csv',index=False)
print('ok')