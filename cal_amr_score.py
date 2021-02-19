import amrlib
from amrlib.graph_processing.amr_plot import AMRPlot
from amrlib.graph_processing.amr_loading import load_amr_entries
from amrlib.evaluate.smatch_enhanced import compute_scores, get_entries, smatch_scores_from_entries, \
    get_cand_ref_entries_from_df, smatch_scores_splitted, smatch_scores_single_str
# from amrlib.evaluate.smatch_enhanced import compute_scores, get_entries
import pandas as pd
import re
import os


# from wodeutil.nlp.metrics.file_util import load_df,get_file_names,get_files_in_dir

def load_df(csv_path=None, ignore_idx=False):
    if csv_path:
        if ignore_idx:
            df = pd.read_csv(csv_path, index_col=[0])
        else:
            df = pd.read_csv(csv_path)
        return df


def get_file_names(file_dir=None, ext='.csv'):
    if file_dir:
        files = os.listdir(file_dir)
        if files:
            filenames = []
            for filename in files:
                if filename:
                    # TODO: can use os function to split the ext and filename
                    # but use re might be more flexible if need other modifications in the future
                    filename = re.sub(ext, '', filename)
                    filenames.append(filename)
            return files, filenames


def get_files_in_dir(file_dir=None, sort_by_name=False):
    if file_dir:
        files = os.listdir(file_dir)
        if files:
            if sort_by_name:
                files = sorted(files)
            return files


def clean_summary(seq, clean_sep=False, sep_start='<t>', sep_end='</t>'):
    if seq:
        seq = re.sub(r'\n', '', seq)  # remove newline character
        seq = re.sub(r'\t', '', seq)
        if clean_sep:
            seq = re.sub(sep_start, '', seq)
            seq = re.sub(sep_end, '', seq)
        seq = seq.strip()
        return seq


def cal_smatch():
    bart_out_path = "/Users/jackz/Google_Drive/GoogleDrive/MyRepo/amrlib/data/abs_amr_single_str/bart_out_abs_with_amr_single_str.csv"
    dest_path = "/Users/jackz/Google_Drive/GoogleDrive/MyRepo/amrlib/data/abs_amr_score_single_str/bart_out_abs.csv"
    df = load_df(csv_path=bart_out_path)
    print('---------- start get cand ref entries ---------------')
    cand_entries, ref_entries = get_cand_ref_entries_from_df(df)
    print('---------- finish get cand ref entries ---------------')
    smatch_precision = 'smatch_precision'
    smatch_recall = 'smatch_recall'
    smatch_f = 'smatch_f_score'
    if cand_entries and ref_entries:
        print('---------- start cal cand ref smatch scores ---------------')
        p_list, r_list, f_list = smatch_scores_single_str(cand_entries, ref_entries)
        print('---------- finish cal cand ref smatch scores ---------------')
        if p_list and r_list and f_list:
            df[smatch_precision] = p_list
            df[smatch_recall] = r_list
            df[smatch_f] = f_list
            df.to_csv(dest_path)
            print('---------- saved new df ---------------')


def cal_smatch_splitted():
    bart_out_path = "/Users/jackz/Google_Drive/GoogleDrive/MyRepo/amrlib/data/abs_amr_splitted/bart_out_abs_with_amr_splitted.csv"
    dest_path = "/Users/jackz/Google_Drive/GoogleDrive/MyRepo/amrlib/data/abs_amr_score_splitted/bart_out_abs_amr_score_splitted.csv"
    df = load_df(csv_path=bart_out_path)
    print('---------- start get cand ref splitted entries ---------------')
    cand_entries, ref_entries = get_cand_ref_entries_from_df(df, sent_splitted=True)
    print('---------- finish get cand ref splitted entries ---------------')
    smatch_precision = 'smatch_precision'
    smatch_recall = 'smatch_recall'
    smatch_f = 'smatch_f_score'
    if cand_entries and ref_entries:
        print('---------- start cal cand ref splitted smatch scores ---------------')
        p_list, r_list, f_list = smatch_scores_splitted(cand_entries, ref_entries)
        print('---------- finish cal cand ref splitted smatch scores ---------------')
        if p_list and r_list and f_list:
            df[smatch_precision] = p_list
            df[smatch_recall] = r_list
            df[smatch_f] = f_list
            df.to_csv(dest_path)
            print('---------- saved new df ---------------')


def example_cal_splitted():
    amrlib_dir = "/content/drive/MyDrive/GoogleDrive/MyRepo/amrlib"
    bart_out_abs_splitted = "/Users/jackz/Google_Drive/GoogleDrive/MyRepo/amrlib/data/abs_amr_splitted/bart_out_abs_with_amr_splitted.csv"
    dest_dir = "/Users/jackz/Google_Drive/GoogleDrive/MyRepo/amrlib/data/abs_amr_score_splitted/bart_out_abs_amr_score_splitted.csv"
    df = load_df(csv_path=bart_out_abs_splitted)
    print('---------- start get cand ref splitted entries ---------------')
    cand_entries, ref_entries = get_cand_ref_entries_from_df(df, sent_splitted=True)
    print('---------- finish get cand ref splitted entries ---------------')
    smatch_precision = 'smatch_precision'
    smatch_recall = 'smatch_recall'
    smatch_f = 'smatch_f_score'
    if cand_entries and ref_entries:
        print('---------- start cal cand ref splitted smatch scores ---------------')
        p_list, r_list, f_list = smatch_scores_splitted(cand_entries, ref_entries)
        print('---------- finish cal cand ref splitted smatch scores ---------------')
        if p_list and r_list and f_list:
            df[smatch_precision] = p_list
            df[smatch_recall] = r_list
            df[smatch_f] = f_list
            df.to_csv(dest_dir)
            print('---------- saved new df ---------------')


def colab_cal_splitted():
    amrlib_dir = "/content/drive/MyDrive/GoogleDrive/MyRepo/amrlib/"
    bart_out_splitted_file = "data/abs_amr_splitted/bart_out_abs_with_amr_splitted.csv"
    src_path = f"{amrlib_dir}{bart_out_splitted_file}"
    dest_bart_out_splitted_score_file = "data/abs_amr_score_splitted/bart_out_abs_amr_score_splitted.csv"
    dest_path = f"{amrlib_dir}{dest_bart_out_splitted_score_file}"
    # bart_out_abs_splitted="/Users/jackz/Google_Drive/GoogleDrive/MyRepo/amrlib/data/abs_amr_splitted/bart_out_abs_with_amr_splitted.csv"
    # dest_dir="/Users/jackz/Google_Drive/GoogleDrive/MyRepo/amrlib/data/abs_amr_score_splitted/bart_out_abs_amr_score_splitted.csv"
    df = load_df(csv_path=src_path)
    print('---------- start get cand ref splitted entries ---------------')
    cand_entries, ref_entries = get_cand_ref_entries_from_df(df, sent_splitted=True)
    print('---------- finish get cand ref splitted entries ---------------')
    smatch_precision = 'smatch_precision'
    smatch_recall = 'smatch_recall'
    smatch_f = 'smatch_f_score'
    if cand_entries and ref_entries:
        print('---------- start cal cand ref splitted smatch scores ---------------')
        p_list, r_list, f_list = smatch_scores_splitted(cand_entries, ref_entries)
        print('---------- finish cal cand ref splitted smatch scores ---------------')
        if p_list and r_list and f_list:
            df[smatch_precision] = p_list
            df[smatch_recall] = r_list
            df[smatch_f] = f_list
            df.to_csv(dest_path)
            print('---------- saved new df ---------------')


def cal_smatch_for_file(df, dest_path='', filename='', sent_splitted=False):
    print(f'---------- start cal smatch score for {filename} ---------------')
    cand_entries, ref_entries = get_cand_ref_entries_from_df(df, sent_splitted=sent_splitted)
    # print('---------- finish get cand ref splitted entries ---------------')
    smatch_precision = 'smatch_precision'
    smatch_recall = 'smatch_recall'
    smatch_f = 'smatch_f_score'
    if cand_entries and ref_entries:
        print('---------- start cal cand ref smatch scores ---------------')
        if sent_splitted:
            p_list, r_list, f_list = smatch_scores_splitted(cand_entries, ref_entries)
        else:
            p_list, r_list, f_list = smatch_scores_single_str(cand_entries, ref_entries)
        print('---------- finish cal cand ref smatch scores ---------------')
        if p_list and r_list and f_list:
            df[smatch_precision] = p_list
            df[smatch_recall] = r_list
            df[smatch_f] = f_list
            df.to_csv(dest_path, index=False)
            print('---------- saved new df ---------------')


def cal_smatch_in_dir(src_dir='', dest_dir='', sent_splitted=False):
    """
    calculate smatch scores for all the csv files under src_dir
    """
    files = get_files_in_dir(src_dir)
    if files and dest_dir:
        ext = '.csv'
        for idx, file in enumerate(files):
            fp = os.path.join(src_dir, file)
            df = load_df(fp)
            fname = re.sub(ext, '', file)
            if sent_splitted:
                new_fname = f'{fname}_amr_score_splitted.csv'
            else:
                new_fname = f"{fname}_amr_score_single_str.csv"
            dest_path = os.path.join(dest_dir, new_fname)
            cal_smatch_for_file(df, dest_path=dest_path, filename=file, sent_splitted=sent_splitted)


def cal_smatch_splitted_all():
    amrlib_dir = "/content/drive/MyDrive/GoogleDrive/MyRepo/amrlib/"
    # bart_out_splitted_file="data/abs_amr_splitted/bart_out_abs_with_amr_splitted.csv"
    # src_path=f"{amrlib_dir}{bart_out_splitted_file}"
    # dest_bart_out_splitted_score_file="data/abs_amr_score_splitted/bart_out_abs_amr_score_splitted.csv"
    # dest_path=f"{amrlib_dir}{dest_bart_out_splitted_score_file}"

    splitted_src_dir = f"{amrlib_dir}data/abs_amr_splitted"
    splitted_dest_dir = f"{amrlib_dir}data/abs_amr_score_splitted"

    src_dir = f"{amrlib_dir}data/abs_amr_single_str"
    dest_dir = f"{amrlib_dir}data/abs_amr_score_single_str"

    cal_smatch_in_dir(src_dir=splitted_src_dir, dest_dir=splitted_dest_dir, sent_splitted=True)
    cal_smatch_in_dir(src_dir=src_dir, dest_dir=dest_dir, sent_splitted=False)


def ext_cal_smatch_splitted_all():
    amrlib_dir = "/content/drive/MyDrive/GoogleDrive/MyRepo/amrlib/"
    splitted_src_dir = f"{amrlib_dir}data/ext_amr_splitted"
    splitted_dest_dir = f"{amrlib_dir}data/ext_amr_scores/ext_amr_score_splitted"
    single_src_dir = f"{amrlib_dir}data/ext_amr_single_str"
    single_dest_dir = f"{amrlib_dir}data/ext_amr_scores/ext_amr_score_single_str"
    cal_smatch_in_dir(src_dir=splitted_src_dir, dest_dir=splitted_dest_dir, sent_splitted=True)
    cal_smatch_in_dir(src_dir=single_src_dir, dest_dir=single_dest_dir, sent_splitted=False)


if __name__ == '__main__':
    # example_sent_to_graph()
    # graph_amr()
    # metric_smatch()
    # cal_smatch()
    # cal_smatch_splitted()
    ext_cal_smatch_splitted_all()
