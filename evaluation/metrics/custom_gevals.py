from __future__ import annotations

from typing import Any

from .metric_helpers import safe_metric_init, try_get_case_param, try_get_case_params


INTENT_HELPFULNESS_CRITERIA = """
Danh gia xem actual_output co xu ly dung intent cua nguoi dung,
bao phu cac rang buoc quan trong trong input va conversation history,
trau chuot nhung khong lan man,
co de xuat next step hop ly khi can,
va khi thieu du lieu thi phai noi ro gioi han/hoi lam ro thay vi phong doan.
Neu expected_output yeu cau tra loi metadata truc tiep va retrieval context co du thong tin,
thi tra loi ngan gon, truc tiep la dat; khong bat buoc them cau hoi lam ro.
""".strip()

DOMAIN_GROUNDED_CRITERIA = """
Danh gia xem cau tra loi co bam sat retrieval context trong domain bat dong san va quy hoach.
Phat nang neu bot tu bo sung gia, dien tich, phap ly, vi tri, loai dat,
hoac ket luan qua muc do tin cay khi retrieval context chua du.
""".strip()

CONVERSATIONAL_CONSISTENCY_CRITERIA = """
Danh gia toan bo hoi thoai xem chatbot co nho va duy tri rang buoc da neu,
co cap nhat cau tra loi khi nguoi dung bo sung dieu kien,
co giai thich ly do thay doi goi y,
co hoi lam ro khi du lieu thieu,
va tranh mau thuan giua cac luot.
Neu cau hoi ro rang va du du lieu thi khong bat buoc hoi lam ro,
va voi follow-up cung chu de thi tra loi bo sung truc tiep duoc xem la hop le.
""".strip()


def build_single_turn_gevals(thresholds: dict[str, Any], judge_model: str | None = None) -> list[Any]:
    from deepeval.metrics import GEval

    st = thresholds.get("single_turn", {})
    input_param, output_param = try_get_case_params()

    retrieval_param = try_get_case_param("RETRIEVAL_CONTEXT") or try_get_case_param("CONTEXT")
    expected_param = try_get_case_param("EXPECTED_OUTPUT")

    intent_eval_params: list[Any] = []
    for param in (input_param, output_param, expected_param, retrieval_param):
        if param is not None:
            intent_eval_params.append(param)

    kwargs_common = {
        "threshold": float(st.get("intent_helpfulness_geval", 0.75)),
    }
    if intent_eval_params:
        kwargs_common["evaluation_params"] = intent_eval_params

    domain_eval_params: list[Any] = []
    for param in (input_param, output_param, expected_param, retrieval_param):
        if param is not None:
            domain_eval_params.append(param)

    domain_kwargs = {
        "threshold": float(st.get("intent_helpfulness_geval", 0.75)),
    }
    if domain_eval_params:
        domain_kwargs["evaluation_params"] = domain_eval_params

    return [
        safe_metric_init(
            GEval,
            name="Intent Coverage & Helpfulness",
            criteria=INTENT_HELPFULNESS_CRITERIA,
            model=judge_model,
            **kwargs_common,
        ),
        safe_metric_init(
            GEval,
            name="Domain-grounded Correctness",
            criteria=DOMAIN_GROUNDED_CRITERIA,
            model=judge_model,
            **domain_kwargs,
        ),
    ]


def build_conversational_geval(thresholds: dict[str, Any], judge_model: str | None = None):
    from deepeval.metrics import ConversationalGEval

    conv = thresholds.get("conversation", {})
    return safe_metric_init(
        ConversationalGEval,
        name="Conversational Consistency",
        criteria=CONVERSATIONAL_CONSISTENCY_CRITERIA,
        model=judge_model,
        threshold=float(conv.get("conversational_geval", 0.75)),
        # OpenAI chat completions currently accepts top_logprobs <= 5.
        top_logprobs=5,
    )
