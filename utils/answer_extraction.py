"""
答案提取与匹配工具
支持多种数学答案格式的提取和比较
"""
import re
import math
from typing import Optional


def extract_boxed_answer(text: str) -> Optional[str]:
    """从 \\boxed{} 中提取答案"""
    # 处理嵌套的大括号
    pattern = r'\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}'
    matches = re.findall(pattern, text)
    if matches:
        return matches[-1].strip()
    return None


def extract_last_number(text: str) -> Optional[str]:
    """提取文本中最后一个数字"""
    # 匹配整数、小数、分数、负数
    pattern = r'[-+]?\d*\.?\d+(?:/\d+)?'
    matches = re.findall(pattern, text)
    if matches:
        return matches[-1]
    return None


def extract_hash_answer(text: str) -> Optional[str]:
    """从 #### 格式中提取答案（GSM8K格式）"""
    if "####" not in text:
        return None
    return text.split("####")[-1].strip()


def extract_answer_from_solution(text: str) -> Optional[str]:
    """从solution标签中提取答案"""
    # 尝试多种格式
    # 1. \boxed{}
    ans = extract_boxed_answer(text)
    if ans:
        return ans
    
    # 2. <answer>...</answer>
    match = re.search(r'<answer>\s*(.*?)\s*</answer>', text, re.DOTALL)
    if match:
        return match.group(1).strip()
    
    # 3. |begin_of_solution| ... |end_of_solution|
    match = re.search(r'\|begin_of_solution\|(.*?)\|end_of_solution\|', text, re.DOTALL)
    if match:
        sol_text = match.group(1).strip()
        ans = extract_boxed_answer(sol_text)
        if ans:
            return ans
        return extract_last_number(sol_text)
    
    # 4. 最后一个数字
    return extract_last_number(text)


def normalize_answer(answer: str) -> str:
    """归一化答案格式"""
    if answer is None:
        return ""
    answer = str(answer).strip()
    # 去除多余空白
    answer = re.sub(r'\s+', ' ', answer)
    # 去除 $ 符号
    answer = answer.replace('$', '')
    # 去除 \left \right 修饰符 (e.g., \left( → (, \right) → ))
    answer = re.sub(r'\\left\s*([(\[{|])', r'\1', answer)
    answer = re.sub(r'\\right\s*([)\]}|])', r'\1', answer)
    answer = re.sub(r'\\left\s*\\([{|])', r'\\\1', answer)
    answer = re.sub(r'\\right\s*\\([}|])', r'\\\1', answer)
    # 归一化括号内的空格: ( 3, x ) → (3, x) → (3,x)
    answer = re.sub(r'\(\s+', '(', answer)
    answer = re.sub(r'\s+\)', ')', answer)
    answer = re.sub(r'\[\s+', '[', answer)
    answer = re.sub(r'\s+\]', ']', answer)
    # 去除数字中的千位分隔符逗号 (必须在逗号空格归一化之前)
    # 只匹配逗号后恰好3位数字的情况 (e.g., "2,125" -> "2125", but NOT "3,7")
    while re.search(r'(\d),(\d{3})(?!\d)', answer):
        answer = re.sub(r'(\d),(\d{3})(?!\d)', r'\1\2', answer)
    # 归一化逗号周围空格: ", " → ","
    answer = re.sub(r'\s*,\s*', ',', answer)
    # 去除 \text{} 包裹
    answer = re.sub(r'\\text\{([^}]*)\}', r'\1', answer)
    # 去除 \mathrm{} 包裹
    answer = re.sub(r'\\mathrm\{([^}]*)\}', r'\1', answer)
    # 统一分数格式
    answer = answer.replace('\\frac', '\\frac')
    # 去除尾部句号
    answer = answer.rstrip('.')
    return answer.strip()


def _eval_fraction(s: str) -> Optional[float]:
    """尝试将分数字符串转换为浮点数"""
    # 匹配 \\frac{a}{b}
    match = re.match(r'\\frac\{([^}]+)\}\{([^}]+)\}', s)
    if match:
        try:
            num = float(match.group(1))
            den = float(match.group(2))
            if den != 0:
                return num / den
        except ValueError:
            pass
    # 匹配 a/b
    match = re.match(r'^([-+]?\d+\.?\d*)\s*/\s*([-+]?\d+\.?\d*)$', s)
    if match:
        try:
            num = float(match.group(1))
            den = float(match.group(2))
            if den != 0:
                return num / den
        except ValueError:
            pass
    return None


def answers_match(pred: str, gold: str, tolerance: float = 1e-6) -> bool:
    """比较预测答案和标准答案是否匹配"""
    if pred is None or gold is None:
        return False
    
    pred = normalize_answer(str(pred))
    gold = normalize_answer(str(gold))
    
    # 精确匹配
    if pred == gold:
        return True
    
    # 数值匹配（处理浮点数精度）
    try:
        pred_val = float(pred)
        gold_val = float(gold)
        if math.isclose(pred_val, gold_val, abs_tol=tolerance):
            return True
    except (ValueError, OverflowError):
        pass
    
    # 分数匹配
    pred_frac = _eval_fraction(pred)
    gold_frac = _eval_fraction(gold)
    if pred_frac is not None and gold_frac is not None:
        if math.isclose(pred_frac, gold_frac, abs_tol=tolerance):
            return True
    
    # 尝试将一方的分数转为浮点数对比
    if pred_frac is not None:
        try:
            gold_val = float(gold)
            if math.isclose(pred_frac, gold_val, abs_tol=tolerance):
                return True
        except ValueError:
            pass
    if gold_frac is not None:
        try:
            pred_val = float(pred)
            if math.isclose(pred_val, gold_frac, abs_tol=tolerance):
                return True
        except ValueError:
            pass
    
    return False


def has_proper_format(response: str) -> bool:
    """检查回答是否有正确的推理格式"""
    # 检查是否有思考过程标记
    has_thinking = (
        '<|begin_of_thought|>' in response or
        '<reasoning>' in response or
        'Step ' in response or
        '\\boxed{' in response
    )
    # 检查是否有答案标记
    has_answer = (
        '\\boxed{' in response or
        '<answer>' in response or
        '<|begin_of_solution|>' in response
    )
    return has_thinking or has_answer


if __name__ == "__main__":
    # 测试
    assert extract_boxed_answer("The answer is \\boxed{42}") == "42"
    assert extract_boxed_answer("\\boxed{\\frac{1}{2}}") == "\\frac{1}{2}"
    assert extract_hash_answer("#### 42") == "42"
    assert answers_match("42", "42")
    assert answers_match("42.0", "42")
    assert answers_match("\\frac{1}{2}", "0.5")
    assert not answers_match("41", "42")
    print("所有测试通过！")
