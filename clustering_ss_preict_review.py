import networkx as nx
import numpy as np
import numpy

from clustering_ss_review import clustering_ss_eval, clustering_recall_correct, data_to_adj_review

import time
from xlwt import Workbook
import os

if __name__ == '__main__':
    starttime = time.time()
    var_dict_main = {}


    def auprgraph_range(adj,file_name):
        os.makedirs('result_range_review', exist_ok=True)
        G = nx.Graph(adj)
        print("nodes - " + str(len(adj)) + " edges - " + str(len(G.edges))+" name - "+str(file_name))
        file = open('./result_range_review/' + file_name + ".txt", 'a')
        tao_array = [5, 10, 15, 20, 25]
        theta_array = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        for tao in tao_array:
            for theta in theta_array:
                ratio = []
                aupr = []
                recall = []
                auc = []
                avg_prec = []
                acc_score = []
                starttime_single = time.time()
                for i in np.arange(0.5, 1, 0.1):  # range is the fraction of edge values included in the graph
                    print(
                        "nodes - " + str(len(adj)) + " edges - " + str(len(G.edges)) + " name - " + str(file_name))
                    print("For tao : ", tao)
                    print("For theta : ", theta)
                    print("For ratio : ", i - 1)
                    avg_range_return = avg_range(G, file_name, tao, theta, i)
                    aupr.append(avg_range_return[0])
                    recall.append(avg_range_return[1])
                    auc.append(avg_range_return[2])
                    avg_prec.append(avg_range_return[3])
                    acc_score.append(avg_range_return[4])
                    ratio.append(i - 1)
                endtime_single = time.time()
                result = "\nFor tao - "+str(tao)+" and for theta - "+str(theta)+"\nRatio :"+str(ratio)+"\nAUPR : "+ str(aupr) + "\nRecall : " +str(recall)+"\nAUC : "+str(auc)+"\nAvg Prec : "+str(avg_prec)+"\nAccuracy score : "+str(acc_score)+"\nTime : "+str((endtime_single - starttime_single))+"sec\n"
                print(str(result))
                file.write(result)
                # Workbook is created
                wb = Workbook()
                # add_sheet is used to create sheet.
                sheet1 = wb.add_sheet('Sheet 1',cell_overwrite_ok=True)
                sheet1.write(0, 0, 'Ratio')
                sheet1.write(0, 1, 'AUPR')
                sheet1.write(0, 2, 'RECALL')
                sheet1.write(0, 3, 'AUC')
                sheet1.write(0, 4, 'AVG PRECISION')
                sheet1.write(0, 5, 'ACCURACY SCORE')
                for i in range(5):
                    sheet1.write(i + 1, 0, ratio[i])
                    sheet1.write(i + 1, 1, aupr[i])
                    sheet1.write(i + 1, 2, recall[i])
                    sheet1.write(i + 1, 3, auc[i])
                    sheet1.write(i + 1, 4, avg_prec[i])
                    sheet1.write(i + 1, 5, acc_score[i])
                wb.save('./result_range_review/' + file_name + "_"+str(tao)+"_"+str(theta)+".xls")
        file.close()


    def avg_range(g, file_name, tao, theta, ratio) :
        #full graph
        aupr = 0
        recall = 0
        auc = 0
        prec = 0
        acc_score = 0
        loop = 10
        for i in range(loop):
            print("for ratio - "+str(ratio))
            print("iteration number - " +str(i))

            value = clustering_recall_correct(g, file_name, tao, theta, ratio)

            aupr += value[0]
            recall += value[1]
            auc += value[2]
            prec += value[3]
            acc_score += value[4]

        return aupr / loop, recall / loop, auc / loop, prec / loop, acc_score / loop


    def aupgraph_range_control_multiple_dataset () :
        file_name_array_single = ['football']
        for file_name in file_name_array_single :
            ds = './datasets/' + file_name
            adj = data_to_adj_review(ds + '.gml')
            auprgraph_range(adj[2], file_name)

    def cluster_eval_single(adj, file_name):
        os.makedirs('result_cluster_eval_review', exist_ok=True)
        tao_array = [5, 10, 15, 20, 25]
        theta_array = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        G = nx.Graph(adj)
        edges = np.array(list(G.edges))
        nodes = list(range(len(adj)))
        print("nodes - " + str(len(adj)) + " edges - " + str(len(G.edges)) + " name - " + str(file_name))
        np.random.shuffle(edges)
        et = edges
        nonedges = np.array(list(nx.non_edges(G)))
        np.random.shuffle(nonedges)
        loop = 10
        file = open('./result_cluster_eval_review/' + file_name + ".txt", 'a')
        cluster_number_matrix = numpy.zeros((5, 10))
        average_isolability_matrix = numpy.zeros((5, 10))
        external_density_matrix = numpy.zeros((5, 10))
        coverage_matrix = numpy.zeros((5, 10))
        modularity_matrix = numpy.zeros((5, 10))
        for tao in tao_array:
            for theta in theta_array:
                starttime_single = time.time()
                print(
                    "nodes - " + str(len(adj)) + " edges - " + str(len(G.edges)) + " name - " + str(
                        file_name))
                print("For tao : ", tao)
                print("For theta : ", theta)
                cluster_number = 0
                average_isolability = 0
                external_density = 0
                coverage = 0
                modularity = 0
                for i in range(loop):
                    print("iteration number - " + str(i))

                    item = clustering_ss_eval(G,tao,theta,file_name)

                    cluster_number += item[0]
                    average_isolability += item[1]
                    external_density += item[2]
                    coverage += item[3]
                    modularity += item[4]
                print("after cluster_eval")
                endtime_single = time.time()
                result = "\nFor tao - " + str(tao) + " and for theta - " + str(theta) +  "\naverage cluster number = " + str(
                    cluster_number / loop) + "\naverage isolability = " + str(
                    average_isolability / loop) + "\nexternal density = " + str(
                    external_density / loop) + "\ncovergae = " + str(coverage / loop) + "\nmodularity = " + str(
                    modularity / loop) + "\nTime : " + str(
                    (endtime_single - starttime_single)) + "sec\n"
                row_index = tao_array.index(tao)
                col_index = theta_array.index(theta)
                cluster_number_matrix[row_index][col_index] += cluster_number / loop
                average_isolability_matrix[row_index][col_index] += average_isolability / loop
                external_density_matrix[row_index][col_index] += external_density / loop
                coverage_matrix[row_index][col_index] += coverage / loop
                modularity_matrix[row_index][col_index] += modularity / loop
                print(str(result))
                file.write(result)
        file.close()
        # Workbook is created
        wb = Workbook()
        # add_sheet is used to create sheet.
        sheet_cno = wb.add_sheet('CLUSTER_NO',cell_overwrite_ok=True)
        sheet_iso = wb.add_sheet('AVG_ISO',cell_overwrite_ok=True)
        sheet_exd = wb.add_sheet('EXTRN_DENS',cell_overwrite_ok=True)
        sheet_cover = wb.add_sheet('COVERAGE',cell_overwrite_ok=True)
        sheet_mod = wb.add_sheet('MODULARITY',cell_overwrite_ok=True)
        sheet_write_array = [sheet_cno, sheet_iso, sheet_exd, sheet_cover, sheet_mod]
        for sheet_no in range(len(sheet_write_array)) :
            count = 1
            for tao in tao_array :
                tao_label = "Tao = "+str(tao)
                sheet_write_array[sheet_no].write(count,0,tao_label)
                count += 1
            count = 1
            for theta in theta_array :
                theta_label = "Theta = "+str(theta)
                sheet_write_array[sheet_no].write(0, count, theta_label)
                count += 1
            for i in range(5):
                for j in range(10):
                    if sheet_no == 0: sheet_write_array[sheet_no].write(i + 1, j + 1, cluster_number_matrix[i][j]/5)
                    if sheet_no == 1: sheet_write_array[sheet_no].write(i + 1, j + 1, average_isolability_matrix[i][j]/5)
                    if sheet_no == 2: sheet_write_array[sheet_no].write(i + 1, j + 1, external_density_matrix[i][j]/5)
                    if sheet_no == 3: sheet_write_array[sheet_no].write(i + 1, j + 1, coverage_matrix[i][j]/5)
                    if sheet_no == 4: sheet_write_array[sheet_no].write(i + 1, j + 1, modularity_matrix[i][j]/5)
        wb.save('./result_cluster_eval_review/' + file_name + ".xls")

    def clustering_check_multiple_dataset():
        file_name_array_single = ['football']

        for file_name in file_name_array_single :
            ds = './datasets/' + file_name
            adj = data_to_adj_review(ds + '.gml')
            cluster_eval_single(adj[2], file_name)


    aupgraph_range_control_multiple_dataset()
    #clustering_check_multiple_dataset()

    endtime = time.time()

    print('That took {} seconds'.format(time.time() - starttime))


