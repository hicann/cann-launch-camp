# Chapter 9锛歀ogSigmoid 鑷畾涔夌畻瀛?

## 椤圭洰淇℃伅

- 鐩綍鍚嶇О锛歚xushuyue_2302_80742797_logsigmoid`
- 浠诲姟鍐呭锛氬畬鎴?LogSigmoid 绠楀瓙鐨?Kernel 渚ф牳鍑芥暟涓?Host 渚?Tiling 鍑芥暟
- 鐩爣纭欢锛欰scend 910B
- 鏀寔杈撳叆/杈撳嚭绫诲瀷锛?
  - `float32`
  - `float16`
  - `bfloat16`

## 鏂囦欢璇存槑

鏈洰褰曞寘鍚互涓嬩笁涓枃浠讹細

1. `README.md`锛氶」鐩瑙堜笌瀹炵幇璇存槑銆?
2. `浠ｇ爜鏂囦欢.md`锛氫笁涓渶瑕佸啓鍏ュ伐绋嬬殑瀹屾暣婧愪唬鐮併€?
3. `杩愯璇存槑.md`锛氫唬鐮佸啓鍏ャ€佹祴璇曡繍琛屽強甯歌鐜拌薄璇存槑銆?

## 绠楀瓙鍔熻兘

LogSigmoid 鐨勮绠楀叕寮忎负锛?

```text
LogSigmoid(x) = log(1 / (1 + exp(-x)))
              = -log(1 + exp(-x))
```

瀹炵幇涓細

- `float32` 鏁版嵁鐩存帴杩涜鍚戦噺璁＄畻銆?
- `float16` 鍜?`bfloat16` 鏁版嵁鍏堣浆鎹负 `float32` 璁＄畻锛屽啀杞崲鍥炲師濮嬬被鍨嬨€?
- Host 渚ф牴鎹緭鍏ユ€诲厓绱犳暟鍜屾暟鎹被鍨嬮€夋嫨鍚堥€傜殑 Block 鏁般€?
- Kernel 渚ф寜 Tile 鍒嗗潡澶勭悊鏁版嵁銆?

## 浠ｇ爜鍐欏叆浣嶇疆

涓変釜婧愭枃浠跺垎鍒啓鍏ワ細

```text
Sources/test/custom_op/op_host/log_sigmoid_custom.cpp
Sources/test/custom_op/op_kernel/log_sigmoid_custom_tiling.h
Sources/test/custom_op/op_kernel/log_sigmoid_custom.cpp
```

瀹屾暣鍐呭瑙?`浠ｇ爜鏂囦欢.md`銆?

## 娴嬭瘯鐢ㄤ緥

| Case | Shape | 鏁版嵁绫诲瀷 |
|---|---|---|
| case1 | `(8, 16, 64)` | `float32` |
| case2 | `(8, 16, 1743)` | `float32` |
| case3 | `(4, 2028)` | `float16` |
| case4 | `(32, 1001, 7763)` | `float16` |
| case5 | `(1, 1024)` | `bfloat16` |
| case6 | `(64, 64, 1024)` | `bfloat16` |

> case4 鏁版嵁閲忓緢澶э紝鐢熸垚杈撳叆鏁版嵁銆丟olden 鏁版嵁浠ュ強鎵ц娴嬭瘯閮藉彲鑳借€楁椂杈冮暱銆?

