# type: ignore
import datasets


def normalize(s: str, full: bool = True):
    l = s.replace("<t> ", "")
    l = l.replace(" </t>", "")
    if full:
        l = l.replace("`` ", "\"")
        l = l.replace(" ''", "\"")
        l = l.replace("` ", "'")
        l = l.replace("-lrb-", "(")
        l = l.replace("-rrb-", ")")
        l = l.replace(" ", "")
    return l


def create_tgt(split: str):
    # Create the map {'lower case' -> sentence list}
    ds = datasets.load_dataset("cnn_dailymail", "3.0.0")
    ds_test = ds[split]
    tgt_map = {}
    for example in ds_test:
        tgt_raw = example["highlights"]
        tgt_split = tgt_raw.split("\n")
        key = tgt_raw.lower()
        key = key.replace(" ", "")

        if key.startswith("thelargehadroncollider"):
            print(repr(key))
        tgt_map[key] = tgt_split

    
    with open("/fsx-meres/tuantran/datasets/sent-summary/cnndm/test.txt.tgt.tagged", encoding="utf-8") as fh:
        output_sents = []
        for idx, line in enumerate(fh):
            line = line.rstrip()
            normalized_sents = [normalize(l) for l in line.split(" </t> <t> ")]
            glge_sents = [normalize(l, full=False) for l in line.split(" </t> <t> ")]
            key = "\n".join(normalized_sents)

            if key not in tgt_map:
                raise ValueError(f"Not found in tgt_map for line {idx+1}: {line} (key = {repr(key)})")
            
            sents = tgt_map[key]
            assert len(glge_sents) == len(sents), f"Different number of sents in {idx+1}: HG sents no. = {len(sents)}, GLGE sents no. = {len(normalized_sents)}"

            txt = []
            for sent, hg_sent in zip(glge_sents, sents):
                i_hg = 0
                for i_glge, char_glge in enumerate(sent):
                    if char_glge.isalnum():
                        assert hg_sent[i_hg].isalnum(), f"Char #{i_hg} is supposed to be an alphanumeric in {hg_sent}, but is '{hg_sent[i_hg]}' (glge char = {char_glge})"

                        if ord("A") <= ord(hg_sent[i_hg]) <= ord("Z"):
                            txt.append(hg_sent[i_hg])
                        else:
                            assert char_glge == hg_sent[i_hg], f"Not matching char: HG sent: {hg_sent}, GLGE sent: {sent} (in index {i_hg}: '{char_glge}' != '{hg_sent[i_hg]}')"
                            txt.append(char_glge)
                        
                        i_hg += 1
                        while i_hg < len(hg_sent) and hg_sent[i_hg] == " ":
                            i_hg += 1

                    elif char_glge not in ["`", "'"]:
                        if char_glge != " ":
                            assert char_glge == hg_sent[i_hg], f"Not matching char: HG sent: {hg_sent}, GLGE sent: {sent} (in index {i_hg}: '{char_glge}' != '{hg_sent[i_hg]}')"
                            i_hg += 1
                            while i_hg < len(hg_sent) and hg_sent[i_hg] == " ":
                                i_hg += 1
                        txt.append(char_glge)
                    
                    # Check for `` , ''
                    else:
                        txt.append(char_glge)
                        if sent[i_glge - 1] == char_glge:
                            assert hg_sent[i_hg] == "\"", f"Expect a double quote at index {i_hg} of \"{hg_sent}\".\nGet: '{hg_sent[i_hg]}')"
                            i_hg += 1
                            while i_hg < len(hg_sent) and hg_sent[i_hg] == " ":
                                i_hg += 1
                        elif hg_sent[i_hg] == char_glge or hg_sent[i_hg] == "'":
                            i_hg += 1
                            while i_hg < len(hg_sent) and hg_sent[i_hg] == " ":
                                i_hg += 1

            
                output_sents.append("".join(txt))
            
        print(" ".join(output_sents))
    

def transform_tgt(output_file: str):
    with (
        open("/fsx-meres/tuantran/datasets/sent-summary/cnndm/test.txt.tgt.tagged", encoding="utf-8") as fh,
        open(output_file, "w") as o,
    ):
        for line in fh:
            line = line.replace("</t> <t>", "<S_SEP>")
            line = normalize(line, full=False)
            o.write(line)


if __name__ == "__main__":
    import fire
    fire.Fire()