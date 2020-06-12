

def conv(folder, prefix = '/home/rizwan/NLPDV/SANCL/POS/gweb_sancl/parse/', file_name='test', suffix='.conll', tag_idx=3):
    path = prefix+"/"+folder+"/"+file_name+suffix
    with open(path, 'r') as rf, open(path+".txt", 'w') as wf:
        for line in rf:
            if line.strip():
                line = line.split()
                wf.write(line[tag_idx]+"\n")
            else:
                wf.write("\n")

ALL_POS_DOMAINS = 'wsj_emails_newsgroups_answers_reviews_weblogs'.split('_')
for folder in ALL_POS_DOMAINS:
    # conv(folder)

    try:
        conv(folder+"_True", prefix='resources/taggers/example-pos/',suffix='.tsv', tag_idx=1)
    except:
        pass
    conv(folder, prefix='resources/taggers/example-pos/', suffix='.tsv', tag_idx=1)