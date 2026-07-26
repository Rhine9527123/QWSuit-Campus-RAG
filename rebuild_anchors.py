"""重建锚点集：n-gram 高频统计 + 子串去重（避免拦腰斩断）

策略：
1. 提取 2~6 字 ngram
2. 过滤停用词（虚词/通用词/无检索意义的词）
3. 频次 ≥ 2 直接保留；频次 = 1 只保留 3 字以上
4. 子串去重：长词优先，剔除其子串（保留"奖学金"剔除"奖学"/"学金"）
"""
import os
import re
import json
from collections import Counter

DATA_DIR = "data"
OUTPUT = os.path.join(DATA_DIR, "anchor_set.json")
TOP_N = 300

# 虚词/通用动词/无检索意义的词
STOP = {
    # 代词/虚词/问句词
    "一个","这个","那个","我们","他们","什么","怎么","可以","没有","不是",
    "已经","还是","因为","所以","但是","如果","而且","或者","不过","虽然",
    "然后","之后","之前","一些","一下","不会","不能","不要","应该","可能",
    "需要","问题","回答","请问","帮我","我想","我要","关于","这是","那是",
    "这些","那些","这样","那样","的话","的吗","了呢","了吗","有的",
    # 通用动词/形容词/名词
    "根据","具体","直接","进行","相关","其他","主要","建议","安排","流程",
    "方式","服务","使用","时间","位置","地点","分布","分配","条件","标准",
    "等级","计算","优先","优惠","灵活","加入","申请","预约","咨询","充值",
    "打印","公布","通知","为准","完成","参加","处理","情况","要求","规则",
    "调整","选择","途径","重要","限制","查看","面试","组织","缴费","每个",
    "每年","当天","地址","区域","价格","收费","免费","类型","职位","职责",
    "资源","设施","设有","详见","自行","基本","准备","保密","支持","接入",
    "每天","每周","每月","每学期","每次","当天","通过","进入","进出","靠近",
    "学校","校园","学生","同学","大学","学院","信息","技术","职业","专业",
    "新生","报到","入学","毕业","就业","教学","上课","下课","自习","考勤",
    "考试","成绩","学分","学时","学籍","课表","课程","课时","选课","重修",
    "补考","挂科","旷课","期中","期末","理论","实训","军训","竞赛","大赛",
    "实习","校招","大一","大二","大三","在校生","学费","贷款","贫困","社团",
    "志愿","班长","委员","竞选","投票","心理","宿舍","宿管","导员","辅导员",
    "辅导","体育","体育课","本科","专科","开放","小时","分钟","校内","校门",
    "东门","南门","北门","西门","复印","复印件","公布","内部","深圳",
    # 数字/编号
    "十一","十二","十三","十四","十五","十六","十七","十八","十九","二十",
    "三十","四十","今日","当年","上报","入围","病假","事假","请假",
    # 短连接/虚词组合
    "与学","与电","与日","与就业","与日常","与生活",
}


def extract_candidates(text: str) -> Counter:
    """提取 2~6 字中文 ngram + 3 字以上英文/数字 token"""
    cjk_pat = re.compile(r"[\u4e00-\u9fff]+")
    alnum_pat = re.compile(r"[a-zA-Z0-9]+")
    counter = Counter()

    for m in cjk_pat.finditer(text):
        seg = m.group()
        for n in range(2, 7):
            if len(seg) >= n:
                for i in range(len(seg) - n + 1):
                    counter[seg[i:i+n]] += 1

    for m in alnum_pat.finditer(text):
        tok = m.group()
        if len(tok) >= 3:
            counter[tok.lower()] += 1

    return counter


def dedupe_substrings(words_with_freq, top_n):
    """
    子串去重 v2：频次优先 + 父串替换
    - 按 频次倒序→长度倒序 排序
    - 长词(≥3字)加入时，从短词池移除其所有子串
    - 短词(2字)加入时，检查是否是已保留长词的子串
    """
    sorted_words = sorted(
        words_with_freq,
        key=lambda x: (-x[1], -len(x[0]))
    )
    kept_long = []   # 长词池（≥3字）
    kept_short = []  # 短词池（2字）

    for word, freq in sorted_words:
        if len(word) >= 3:
            # 长词：是否是已保留长词的子串
            if any(word in k for k in kept_long):
                continue
            # 从短词池移除其所有子串
            kept_short = [k for k in kept_short if k not in word]
            kept_long.append(word)
        else:
            # 短词：是否是已保留长词的子串
            if any(word in k for k in kept_long):
                continue
            if word in kept_short:
                continue
            kept_short.append(word)

        if len(kept_long) + len(kept_short) >= top_n:
            break

    return kept_long + kept_short


# 实体后缀：自动识别低频但重要的实体词
ENTITY_SUFFIXES = ("处", "室", "馆", "楼", "门", "院", "部", "局", "科")

# 重要实体白名单：即使文档频次低也强制保留（用于路由命中）
ENTITY_WHITELIST = {
    "警务处", "教务处", "保卫处", "医务室", "全称", "运动场",
}


def main():
    print("=" * 60)
    print("重建锚点集：n-gram 高频统计 + 子串去重")
    print("=" * 60)

    # 1. 读取所有文档
    all_text = ""
    file_count = 0
    for fn in sorted(os.listdir(DATA_DIR)):
        if fn.endswith(".txt"):
            with open(os.path.join(DATA_DIR, fn), encoding="utf-8") as f:
                all_text += f.read() + "\n"
                file_count += 1
            print(f"  读取: {fn}")
    print(f"共读取 {file_count} 篇文档，{len(all_text)} 字符")

    # 2. 提取候选词频次
    counter = extract_candidates(all_text)
    print(f"候选 ngram 数: {len(counter)}")

    # 3. 过滤停用词
    for s in list(counter.keys()):
        if s in STOP:
            del counter[s]

    # 4. 频次阈值
    high_freq = [(w, c) for w, c in counter.items() if c >= 2]
    low_freq = [(w, c) for w, c in counter.items() if c == 1 and len(w) >= 3]
    candidates = high_freq + low_freq
    print(f"高频(>=2): {len(high_freq)} | 低频(=1, >=3字): {len(low_freq)} | 总候选: {len(candidates)}")

    # 5. 子串去重
    final_words = dedupe_substrings(candidates, TOP_N)
    print(f"去重后保留 {len(final_words)} 个锚点")

    # 5b. 强制补入实体白名单 + 实体后缀词（确保关键查询能命中路由）
    forced = []
    for w in ENTITY_WHITELIST:
        if w not in final_words:
            final_words.append(w)
            forced.append(w)

    # 自动识别实体后缀词（频次≥1且以"处/室/馆/楼/门/院/部"结尾的3字以上词）
    # 过滤：剔除"出/进/到/近/扫/管/馆"等动词前缀造成的拦腰斩断
    BAD_PREFIXES = ("出", "进", "到", "近", "扫", "过", "送", "配", "按", "从",
                    "去", "管", "馆", "楼", "院", "处", "室", "到", "是", "或",
                    "员", "士", "知", "再", "段", "育", "果", "分", "保", "以",
                    "他", "学", "委", "园", "细", "副", "育", "定", "圳", "校",
                    "班", "体", "如", "的")
    for w, c in counter.items():
        if (len(w) >= 3 and w.endswith(ENTITY_SUFFIXES)
                and w not in final_words
                and not w[0] in BAD_PREFIXES  # 首字符不能是动词
                and not any(w in k for k in final_words if len(k) > len(w))):
            final_words.append(w)
            forced.append(w)

    if forced:
        print(f"强制补入 {len(forced)} 个实体词: {forced}")

    # 6. 展示 Top-30 高频词
    print("\nTop-30 高频候选词:")
    for w, c in sorted(candidates, key=lambda x: -x[1])[:30]:
        in_set = "OK " if w in final_words else "   "
        print(f"  {in_set} {w:12s} freq={c}")

    # 7. 保存
    data = {
        "anchor_set": sorted(final_words),
        "total_docs_scanned": file_count,
        "pending_doc_count": 0,
        "pending_top_words": [],
        "version": "3.0-ngram-dedup",
    }
    with open(OUTPUT, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"\n[OK] 已保存到 {OUTPUT}")

    # 8. 关键查询词检查
    print("\n关键查询词检查:")
    test_words = ["警务处", "医务室", "运动场", "转专业", "全称",
                  "食堂", "奖学金", "图书馆", "体育馆",
                  "教务处", "校医院", "电话", "保卫处", "运动场所"]
    for w in test_words:
        in_set = "OK  " if w in final_words else "MISS"
        doc_freq = counter.get(w, 0)
        print(f"  {in_set} {w:10s} (doc_freq={doc_freq})")


if __name__ == "__main__":
    main()
