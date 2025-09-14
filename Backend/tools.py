import requests

def web_search(query: str, num_results: int):
    """
    Tìm kiếm thông tin trên internet bằng SearXNG.
    Nếu khách hàng hỏi thông tin về Microsoft Office Specialist (MOS) thì sẽ trả về kết quả từ SearXNG.
    
    Args:
        query (str): Chuỗi truy vấn tìm kiếm.
        num_results (int): Số lượng kết quả trả về.
        dict: Danh sách kết quả (title, url, snippet).
    """
    base_url="http://localhost:8080"
    url = f"{base_url}/search"
    params = {
        "q": query,  # dùng json_raw thay cho json
        "language": "vi",
        "safesearch": 1,
        "format": "json",
        "num_results": num_results,  # Số lượng kết quả trả về  # Chọn engine tìm kiếm, có thể là google, bing, duckduckgo, v.v.
    }
    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; LLMFramework/1.0)"
    }
    
    res = requests.get(url, params=params, headers=headers, timeout=15)
    print("Status:", res.status_code)
    print("Headers:", res.headers.get("content-type"))
    print("Body preview:", res.text[:500])   # in thử 500 ký tự đầu
    res.raise_for_status()
    return res.json()