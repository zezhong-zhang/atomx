# Actual 22:00 HKT hourly delivery

The hourly service sent message `om_x100b638d288ee13cc4302107d4fb072`
to the authorized current private chat at 2026-10-11 22:00:04 HKT. A single
inherited-bridge `lark-cli im +messages-mget` call completed with exit 0;
the readback confirms the bot identity, chat, message, interactive type,
and position 1607. [Open the actual message](https://applink.feishu.cn/client/chat/open?openChatId=oc_fb2fc627b3e98dc9555de31580878758&position=1607).

[delivery_proof.json](delivery_proof.json) binds the actual
[card payload](card.json), [receipt](receipt.json),
[service status](send_status.json), [readback](readback.json), and
[readback execution](readback_execution.json). The sent payload uses
CardKit schema 2.0 and has 15 markdown elements. The CLI returns rendered
card text rather than raw CardKit DSL: all visible content agrees, with
six folded newline characters and one automatic DOI link wrapper. The
proof records these differences without claiming raw readback DSL equality.

This delivered card binds stage016 SHA
`247ba573bf265c23254760f39df16c3517012999cade1c7d4c898b2d9727eccb`
and registry SHA
`ade386ad9689c1a8ef054d22ec55893f3468830986aa34a580169ac5a5a42bf1`.
[registry_stage016.json](registry_stage016.json) was copied from the
immutable stage017 `before_progress.json` after the live registry advanced;
the delivered source identity is preserved.

The actual same-round [iteration request](iteration_request.json),
[preview](iteration_preview.json), and [decision](iteration_decision.json)
have status `implementation_frontier`, `job: null`, and no service jobs.
Successful dispatch means that the iteration request was registered. It
does not mean that the hourly service automatically executed physics.

Root separately continued the authorized iteration with the bounded native
normalized-density gradient control. Its stage017 report, SHA
`395b7dd64dedba5d71b16c56adc75f36394250ec7561527f8acdcf8cd40784c4`,
records closure with 1 H, 1 CI, 2 weighted gradients, 24.946 seconds, and
104 live checks passed. This delivery audit performs zero scientific calls
and does not independently certify that numerical result. Original
derivative failures, stationarity failures, and optimizer HOLD remain in scope.

Only actual card, receipt, request, decision, status, readback, and the
source registry preimage were copied. No configuration or credentials were
read or copied. [verify_delivery.py](verify_delivery.py) verifies archived
local evidence and makes no API or science calls.
