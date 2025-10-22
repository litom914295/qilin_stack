
# 执行端验签示例
## CSV
- 校验 `orders.csv` 的 SHA256 是否与 `orders.csv.sig` 一致。

## HTTP（服务端伪代码）
```python
import hmac, hashlib, time
def verify(req_body: bytes, headers: dict, api_secret: str) -> bool:
    ts = int(headers.get("X-Timestamp","0"))
    if abs(time.time() - ts) > 60: return False
    nonce = headers.get("X-Nonce","")
    sign = headers.get("X-Signature","")
    msg = "|".join([str(ts), nonce, req_body.decode("utf-8")]).encode("utf-8")
    expect = hmac.new(api_secret.encode("utf-8"), msg, hashlib.sha256).hexdigest()
    return hmac.compare_digest(sign, expect)
```
