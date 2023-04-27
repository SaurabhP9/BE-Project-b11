from googleapiclient.discovery import build
from pprint import pprint
import json

count = 1009

dictionary = {
    "videoId": "xyz",
    "author": " saurabh ",
    "comment_text": "I am a 21 year developer with experiance of 2 months",
    "publish_at": "2022",
}

CHANNEL_ID = "UCPDis9pjXuqyI7RYLJ-TTSA"


def get_youtube():
    DEVELOPER_KEY = "Google Developer Key "
    YOUTUBE_API_SERVICE_NAME = "youtube"
    YOUTUBE_API_VERSION = "v3"
    youtube = build(
        YOUTUBE_API_SERVICE_NAME,
        YOUTUBE_API_VERSION,
        developerKey=DEVELOPER_KEY,
        # maxResults = 50
    )
    return youtube


# def search_video():
youtube = get_youtube()
request = youtube.search().list(
    part="snippet", type="video", channelId="UC-lHJZR3Gqxm24_Vd_AJ5Yw"
)
response = request.execute()
# pprint(response)
video_ids = []

for item in response["items"]:
    title = item["snippet"]["title"]
    videoId = item["id"]["videoId"]
    video_ids.append(videoId)
    request = youtube.commentThreads().list(part="snippet", videoId=videoId)
    response = request.execute()
    for itemsa in video_ids:
        for item in response["items"]:
            comment = item["snippet"]["topLevelComment"]
            author = comment["snippet"]["authorDisplayName"]
            comment_text = comment["snippet"]["textDisplay"]
            # video_id = video["video_id"]
            # video_title = video["video_title"]
            print("author is ", author)
            print("Comment passed is ", comment_text)
            print("                 ")
            print("video id ", itemsa)
            print(" 1 AUTHORS COMMENT IS FINISH")
            dictionary[count] = {
                "videoId": videoId,
                "author": author,
                "comment_text": comment_text,
                # "publish_at": snippet.publishedAt(2012),
            }
            count = count + 1
    print(
        "video id",
        itemsa,
        "is over *****************************************************************************",
    )
    print("/n'saurabh ")

print("********************&&&&&&&&&&&", video_ids, "&&&&&&&&&&***********")


json_object = json.dumps(dictionary)

with open("sample11.json", "w") as outfile:
    outfile.write(json_object)

    # {"category": "comedy", "id": "UC-lHJZR3Gqxm24_Vd_AJ5Yw", "name": "PewDiePie", sample.json},
    # { "category":"comedy", "id":"UC4-79UOlP48-QNGgCko5p2g", "name":"Mrbeast2", sample2.json},

    # {"category": "comedy", "id": "UCY30JRSgfhYXA6i6xX1erWg", "name": "SMOSH"},
    # {"category": "comedy", "id": "UCPDXXXJj9nax0fr0Wfc048g", "name": "CollegeHumor"},
    # {"category": "comedy", "id": "UCPDis9pjXuqyI7RYLJ-TTSA", "name": "FailArmy"},
    # {"category": "comedy", "id": "UC9gFih9rw0zNCK3ZtoKQQyA", "name": "JennaMarbles"},

    # {"category": "tv", "id": "UC8-Th83bH_thdKZDJCrn88g", "name": "Fallon"},
    # {"category": "tv", "id": "UCi7GJNg51C3jgmYTUwqoUXA", "name": "Conan"},
    # {"category": "tv", "id": "UCJ0uqCI0Vqr2Rrt1HseGirg", "name": "Corden"},

    # {"category": "tv", "id": "UCa6vGFO9ty8v5KZJXQxdhaw", "name": "Kimmel" , sample3.json},
    # {"category": "tv", "id": "UCp0hYYBW6IMayGgR-WeoCvQ", "name": "Ellen" , sample4.json},

    # {"category": "science", "id": "UCC552Sd-3nyi_tk2BudLUzA", "name": "AsapSCIENCE"},
    # {"category": "science", "id": "UCHnyfMqiRRG1u-2MsSQLbXA", "name": "Veritasium"},

    # {"category": "science", "id": "UCZYTClx2T1of7BRZ86-8fow", "name": "SciShow", sample5.json},
    # {"category": "science", "id": "UCoxcjq-8xIDTYp3uz647V5A", "name": "Numberphile", sample6.json, mathvideo},
    # {"category": "science", "id": "UCvJiYiBUbw4tmpRSZT2r1Hw", "name": "ScienceChannel", sample7.json},

    # {"category": "news", "id": "UC1yBKRuGpC1tSM73A0ZjYjQ", "name": "TYT"},
    # {"category": "news", "id": "UCBi2mrWuNuyYy4gbM6fU18Q", "name": "ABCNews", sample8.json},
    # {"category": "news", "id": "UCupvZG-5ko_eiXAupbDfxWw", "name": "CNN" , sample10.json},
    # {"category": "news", "id": "UCvsye7V9psc-APX6wV1twLg", "name": "AlexJones"},
    # {"category": "news", "id": "UCLXo7UDZvByw2ixzpQCufnA", "name": "Vox", sample9.json},
