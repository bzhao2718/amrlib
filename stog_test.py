import amrlib
from amrlib.graph_processing.amr_plot import AMRPlot
from amrlib.graph_processing.amr_loading import load_amr_entries
from amrlib.evaluate.smatch_enhanced import compute_scores, get_entries, smatch_scores_from_entries


def example_sent_to_graph():
    stog = amrlib.load_stog_model()
    # graphs = stog.parse_sents(
    #     ['This is a test of the system.', 'This is a second sentence.', 'I like tennis', 'I like to play tennis'],
    #     add_metadata=True)
    cnn_raw = '<t> Audrey Alexander wanted her neighbours to chop down their huge hedge . </t> <t> She claims the 40 ft leylandii was blocking sunlight from reaching her home . </t> <t> Feud started in 1980 when it blocked light from reaching a vegetable patch . </t> <t> Council finally rules that the hedge can stay - but must be cut back to 20 ft   . </t>'
    cnn_processed = 'Audrey Alexander wanted her neighbours to chop down their huge hedge . \
     She claims the 40 ft leylandii was blocking sunlight from reaching her home . \
     Feud started in 1980 when it blocked light from reaching a vegetable patch . \
     Council finally rules that the hedge can stay - but must be cut back to 20 ft.'
    craw = "<t> Rory McIlroy battled with Fifty Shades of Grey star Jamie Dornan to promote new football concept Circular Soccer . </t> <t> The World No 1 golfer defeated the film star 2 - 1 to take the crown , with the deciding goal coming from an impressive long - range finish . </t> <t> The 25-year - old is an avid Manchester United fan and dreamed of playing for the Red Devils as a child . </t>"
    cproc = "<t> Rory McIlroy faced with Fifty Shades of Grey 's Jamie Dornan . </t>  <t> McIlroy and Dornan took part in the first Circular Soccer Showdown of 2015 . </t>  <t> McIlroy finished strongly to finish an impressive fourth at the Masters . </t>  <t> READ : It wo n't be too long before McIlroy wins a Masters . </t>"
    cnn_graphs = stog.parse_sents([craw, cproc], add_metadata=True)
    with open('amrlib/data/example/cnn_cand_91.txt', 'w') as f:
        for cnn in cnn_graphs:
            f.write(cnn + '\n\n')
            print(cnn)
    # for graph in graphs:
    #     print(graph)


def graph_amr():
    input_file = 'amrlib/data/example/out_cand.pred.post'
    # Load the AMR file
    entries = load_amr_entries(input_file)
    entry = entries[4]  # pick an index
    # Plot
    plot = AMRPlot()
    plot.build_from_graph(entry, debug=False)
    plot.view()


def metric_smatch():
    GOLD = 'amrlib/data/example/out_cand.pred.post'
    PRED = 'amrlib/data/example/out_cand.pred.post'
    cnn1 = 'amrlib/data/example/cnn_ref.txt'
    gold_entries = get_entries(cnn1)
    pred_entries = get_entries(cnn1)

    compute_scores(cnn1, cnn1)
    print('-------------------------')
    smatch_scores_from_entries(pred_entries, gold_entries)


if __name__ == '__main__':
    # example_sent_to_graph()
    # graph_amr()
    metric_smatch()
