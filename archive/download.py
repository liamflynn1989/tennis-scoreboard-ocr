import yt_dlp

ydl_opts = {
    "format": "bv*[height<=720]+ba/b[height<=720]",
    "merge_output_format": "mp4",
    "force_ipv4": True,
    "cookiefile": "cookies.txt",
    "http_headers": {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        )
    },
}


def download_video(video_url):
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([video_url])
    return "Download completed."

if __name__ == "__main__":
    url = "https://www.youtube.com/watch?v=ATI9B7ZLof8&t"
    url = 'https://www.youtube.com/watch?v=I2napxp1ym0'
    download_video(url)