# Projeto de desenvolvimento: Embeddings de Grafos Financeiros

Este projeto faz parte da disciplina **Engenharia de Aprendizado de Máquina (CCF726)** do **Mestrado em Ciência da Computação – UFV (Campus Florestal)**. A proposta é aplicar, ponta-a-ponta, o ciclo de ML em um problema real com grafos: da definição e preparação dos dados à modelagem, avaliação e artefatos de implantação.

## Contexto: representações em grafos e Graph Foundation Models (GFMs)

Muitos problemas do mundo real são relacionais: transações entre contas, interações sociais, cadeias de suprimento. Nesses cenários, o desempenho do modelo depende de como representamos nós, arestas e o próprio grafo.
Historicamente, o caminho evoluiu de:

* **Features manuais (grau, centralidades, contagens)**: informativas porém limitadas;

* **Embeddings não supervisionados (DeepWalk/Node2Vec)**: capturam proximidade/estrutura com caminhadas aleatórias;

* **GNNs (GraphSAGE/GIN/GAT)**: aprendem representações end-to-end via message passing, combinando estrutura + atributos.

Atualmente, uma das formas de gerar essas representações são os **Graph Foundation Models (GFMs)**: modelos pré-treinados em grandes coleções de grafos, que aprendem representações gerais (de nós/arestas/subgrafos/grafos) e depois são adaptadas (zero-/few-shot, fine-tune, adapters) para domínios como finanças (fraude/AML, crédito, risco, anomalias) ou outros domínios (por isso são modelos fundacionais e generalistas).

Minha pesquisa de mestrado está nesse contexto: GFMs para grafos do contexto financeiro, visando a gereação de embeddings robustos e representativos que sirvam para serem aplicados em diferentes tarefas como, por exemplo, detecção de fraudes, análise de crédito, detecção de anomalias, etc.

## Objetivo desta disciplina

Na disciplina, a meta prática foi consolidar esse conhecimento sobre representações em grafos construindo um pipeline reprodutível para classificar transações como lícitas/ilícitas, partindo de uma arquitetura de GNNs já consolidade de forma também a compreender mais tecnicamente o processo de geração de embeddings em grafos. O projeto cobre:

* Construção do grafo a partir de transações;

* Engenharia de atributos de nós e arestas sensíveis ao tempo;

* Baselines fortes (tabular/Node2Vec) e um modelo GraphSAGE+MLP proposto;

* Avaliação justa com split temporal e threshold tuning;

* Empacotamento/artefatos para aplicação e reuso do modelo.

## Definição do problema

> Dado um histórico de transações entre contas, classificar se uma transação é ilícita (classe 1) ou lícita (classe 0), usando:

* Atributos da aresta (valor, par de moedas, formato de pagamento, tempo desde última transação, rollings/EWMs do remetente/destinatário etc.);

* Contexto estrutural (vizinhança no grafo via GNN/Node2Vec);

* Split temporal para evitar vazamento (treino → validação → teste por quantis do timestamp).

Métricas-chave: **F1**, **Precisão**, **Recall** (comércio costuma preferir alto recall para reduzir FN), além de **AUROC**/**AUPRC**. A decisão final usa limiar (thr) calibrado na validação.

## Base de dados utilizada

**IBM Transactions for Anti-Money Laundering (AML)**

Kaggle: *ealtman2019/ibm-transactions-for-anti-money-laundering-aml*

Informações gerais:

> * ~5,08 milhões de transações.
> * Colunas principais: `timestamp`, `src_account`, `dst_account`, `amount_paid/received`, `pay/recv_currency`, `payment_format`, `label (0/1)`, `fx_spread`, etc.
> * Desbalanceamento forte representando o mundo real (ilícitas ≪ lícitas): utilizado class_weight e tuning de limiar para lidar com essa característica.

## Referências

- Perozzi, B., Al-Rfou, R., & Skiena, S. (2014). **DeepWalk: Online Learning of Social Representations**. *KDD*. [doi:10.1145/2623330.2623732](https://doi.org/10.1145/2623330.2623732)
- Grover, A., & Leskovec, J. (2016). **node2vec: Scalable Feature Learning for Networks**. *KDD*. [arXiv:1607.00653](https://arxiv.org/abs/1607.00653)
- Hamilton, W., Ying, Z., & Leskovec, J. (2017). **Inductive Representation Learning on Large Graphs (GraphSAGE)**. *NeurIPS*. [arXiv:1706.02216](https://arxiv.org/abs/1706.02216)
- Veličković, P., Cucurull, G., Casanova, A., Romero, A., Liò, P., & Bengio, Y. (2018). **Graph Attention Networks (GAT)**. *ICLR*. [arXiv:1710.10903](https://arxiv.org/abs/1710.10903)
- Xu, K., Hu, W., Leskovec, J., & Jegelka, S. (2019). **How Powerful are Graph Neural Networks? (GIN)**. *ICLR*. [arXiv:1810.00826](https://arxiv.org/abs/1810.00826)
- Veličković, P., et al. (2019). **Deep Graph Infomax (DGI)**. *ICLR*. [arXiv:1809.10341](https://arxiv.org/abs/1809.10341)
- Hu, W., Liu, B., Gomes, J., Zitnik, M., Liang, P., Pande, V., & Leskovec, J. (2020). **Strategies for Pre-training Graph Neural Networks**. *ICLR*. [arXiv:1905.12265](https://arxiv.org/abs/1905.12265)
- Bommasani, R., et al. (2021). **On the Opportunities and Risks of Foundation Models**. *Stanford CRFM*. [arXiv:2108.07258](https://arxiv.org/abs/2108.07258)
- **IBM Transactions for Anti-Money Laundering (AML)** — *Kaggle dataset*. [Link](https://www.kaggle.com/datasets/ealtman2019/ibm-transactions-for-anti-money-laundering-aml)

