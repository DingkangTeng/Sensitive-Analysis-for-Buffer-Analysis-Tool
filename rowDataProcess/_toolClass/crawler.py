import requests, time, random

from .apiKey import Headers

class crawler:
    __headers = Headers

    def __init__(self, url: str, postData: dict = {}):
        self.url = url
        self.postData = postData

    def rpost(self) -> requests.Response:
        try:
            r = requests.post(self.url, headers=self.__headers, data = self.postData, timeout=(3,7))
            return r
        except:
            print("Network Error")
            time.sleep(random.randint(5,10))
            r = self.rpost()
            return r
    
    def rget(self) -> requests.Response:
        try:
            r = requests.get(self.url, headers=self.__headers, timeout=(3,7))
            return r
        except:
            print("Network Error")
            time.sleep(random.randint(5,10))
            r = self.rget()
            return r