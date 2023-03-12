import numpy as np
import pandas as pd

from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import DBSCAN

#plotting
import matplotlib.pyplot as plt
import seaborn as sns


from tqdm import tqdm, trange
colgroups = [
    ['f190486d6', '58e2e02e6', 'eeb9cd3aa', '9fd594eec', '6eef030c1', '15ace8c9f', 'fb0f5dbfe', '58e056e12', '20aa07010', '024c577b9', 'd6bb78916', 'b43a7cfd5', '58232a6fb', '1702b5bf0', '324921c7b', '62e59a501', '2ec5b290f', '241f0f867', 'fb49e4212', '66ace2992', 'f74e8f13d', '5c6487af1', '963a49cdc', '26fc93eb7', '1931ccfdd', '703885424', '70feb1494', '491b9ee45', '23310aa6f', 'e176a204a', '6619d81fc', '1db387535', 'fc99f9426', '91f701ba2', '0572565c2', '190db8488', 'adb64ff71', 'c47340d97', 'c5a231d81', '0ff32eb98'],
    ['266525925', '4b6dfc880', '2cff4bf0c', 'a3382e205', '6488c8200', '547d3135b', 'b46191036', '453128993', '2599a7eb7', '2fc60d4d9', '009319104', 'de14e7687', 'aa31dd768', '2b54cddfd', 'a67d02050', '37aab1168', '939cc02f5', '31f72667c', '6f951302c', '54723be01', '4681de4fd', '8bd53906a', '435f27009', 'f82167572', 'd428161d9', '9015ac21d', 'ec4dc7883', '22c7b00ef', 'd4cc42c3d', '1351bf96e', '1e8801477', 'b7d59d3b5', 'a459b5f7d', '580f5ff06', '39b3c553a', '1eec37deb', '692c44993', 'ce8ce671e', '88ef1d9a8', 'bf042d928'],
    ['9d5c7cb94', '197cb48af', 'ea4887e6b', 'e1d0e11b5', 'ac30af84a', 'ba4ceabc5', 'd4c1de0e2', '6d2ece683', '9c42bff81', 'cf488d633', '0e1f6696a', 'c8fdf5cbf', 'f14b57b8f', '3a62b36bd', 'aeff360c7', '64534cc93', 'e4159c59e', '429687d5a', 'c671db79e', 'd79736965', '2570e2ba9', '415094079', 'ddea5dc65', 'e43343256', '578eda8e0', 'f9847e9fe', '097c7841e', '018ab6a80', '95aea9233', '7121c40ee', '578b81a77', '96b6bd42b', '44cb9b7c4', '6192f193d', 'ba136ae3f', '8479174c2', '64dd02e44', '4ecc3f505', 'acc4a8e68', '994b946ad'],
    ['f1eeb56ae', '62ffce458', '497adaff8', 'ed1d5d137', 'faf7285a1', 'd83da5921', '0231f07ed', '7950f4c11', '051410e3d', '39e1796ab', '2e0148f29', '312832f30', '6f113540d', 'f3ee6ba3c', 'd9fc63fa1', '6a0b386ac', '5747a79a9', '64bf3a12a', 'c110ee2b7', '1bf37b3e2', 'fdd07cac1', '0872fe14d', 'ddef5ad30', '42088cf50', '3519bf4a4', 'a79b1f060', '97cc1b416', 'b2790ef54', '1a7de209c', '2a71f4027', 'f118f693a', '15e8a9331', '0c545307d', '363713112', '73e591019', '21af91e9b', '62a915028', '2ab5a56f5', 'a8ee55662', '316b978cd'],
    ['b26d16167', '930f989bf', 'ca58e6370', 'aebe1ea16', '03c589fd7', '600ea672f', '9509f66b0', '70f4f1129', 'b0095ae64', '1c62e29a7', '32a0342e2', '2fc5bfa65', '09c81e679', '49e68fdb9', '026ca57fd', 'aacffd2f4', '61483a9da', '227ff4085', '29725e10e', '5878b703c', '50a0d7f71', '0d1af7370', '7c1af7bbb', '4bf056f35', '3dd64f4c4', 'b9f75e4aa', '423058dba', '150dc0956', 'adf119b9a', 'a8110109e', '6c4f594e0', 'c44348d76', 'db027dbaf', '1fcba48d0', '8d12d44e1', '8d13d891d', '6ff9b1760', '482715cbd', 'f81c2f1dd', 'dda820122'],
    ['c928b4b74', '8e4d0fe45', '6c0e0801a', '02861e414', 'aac52d8d9', '041c5d0c9', 'd7875bb6c', 'e7c0cfd0f', 'd48c08bda', '0c9462c08', '57dd44c29', 'a93118262', '850027e38', 'db3839ab0', '27461b158', '32174174c', '9306da53f', '95742c2bf', '5831f4c76', '1e6306c7c', '06393096a', '13bdd610a', 'd7d314edc', '9a07d7b1f', '4d2671746', '822e49b95', '3c8a3ced0', '83635fb67', '1857fbccf', 'c4972742d', 'b6c0969a2', 'e78e3031b', '36a9a8479', 'e79e5f72c', '092271eb3', '74d7f2dc3', '277ef93fc', 'b30e932ba', '8f57141ec', '350473311'],
    ['06148867b', '4ec3bfda8', 'a9ca6c2f4', 'bb0408d98', '1010d7174', 'f8a437c00', '74a7b9e4a', 'cfd55f2b6', '632fed345', '518b5da24', '60a5b79e4', '3fa0b1c53', 'e769ee40d', '9f5f58e61', '83e3e2e60', '77fa93749', '3c9db4778', '42ed6824a', '761b8e0ec', 'ee7fb1067', '71f5ab59f', '177993dc6', '07df9f30c', 'b1c5346c4', '9a5cd5171', 'b5df42e10', 'c91a4f722', 'd93058147', '20a325694', 'f5e0f4a16', '5edd220bc', 'c901e7df1', 'b02dfb243', 'bca395b73', '1791b43b0', 'f04f0582d', 'e585cbf20', '03055cc36', 'd7f15a3ad', 'ccd9fc164'],
    ['df838756c', '2cb73ede7', '4dcf81d65', '61c1b7eb6', 'a9f61cf27', '1af4d24fa', 'e13b0c0aa', 'b9ba17eb6', '796c218e8', '37f57824c', 'd1e0f571b', 'f9e3b03b7', 'a3ef69ad5', 'e16a20511', '04b88be38', '99e779ee0', '9f7b782ac', '1dd7bca9f', '2eeadde2b', '6df033973', 'cdfc2b069', '031490e77', '5324862e4', '467bee277', 'a3fb07bfd', '64c6eb1cb', '8618bc1fd', '6b795a2bc', '956d228b9', '949ed0965', 'a4511cb0b', 'b64425521', '2e3c96323', '191e21b5f', 'bee629024', '1977eaf08', '5e645a169', '1d04efde3', '8675bec0b', '8337d1adc'],
    ['a1cd7b681', '9b490abb3', 'b10f15193', '05f54f417', 'a7ac690a8', 'ed6c300c2', 'd0803e3a1', 'b1bb8eac3', 'bd1c19973', 'a34f8d443', '84ec1e3db', '24018f832', '82e01a220', '4c2064b00', '0397f7c9b', 'ba42e41fa', '22d7ad48d', '9abffd22c', 'dbfa2b77f', '2c6c62b54', '9fa38def3', 'ecb354edf', '9c3154ae6', '2f26d70f4', '53102b93f', 'a36b95f78', '1fa0f78d0', '19915a6d3', 'c944a48b5', '482b04cba', '2ce77a58f', '86558e595', 'c3f400e36', '20305585c', 'f8ccfa064', 'dd771cb8e', '9aa27017e', 'cd7f0affd', '236cc1ff5', 'a3fc511cd'],
    ['920a04ee2', '93efdb50f', '15ea45005', '78c57d7cd', '91570fb11', 'c5dacc85b', '145c7b018', '590b24ab1', 'c283d4609', 'e8bd579ae', '7298ca1ef', 'ce53d1a35', 'a8f80f111', '2a9fed806', 'feb40ad9f', 'cfd255ee3', '31015eaab', '303572ae2', 'cd15bb515', 'cb5161856', 'a65b73c87', '71d64e3f7', 'ec5fb550f', '4af2493b6', '18b4fa3f5', '3d655b0ed', '5cc9b6615', '88c0ec0a6', '8722f33bb', '5ed0c24d0', '54f26ee08', '04ecdcbb3', 'ade8a5a19', 'd5efae759', 'ac7a97382', 'e1b20c3a6', 'b0fcfeab8', '438b8b599', '43782ef36', 'df69cf626'],
    ['50603ae3d', '48282f315', '090dfb7e2', '6ccaaf2d7', '1bf2dfd4a', '50b1dd40f', '1604c0735', 'e94c03517', 'f9378f7ef', '65266ad22', 'ac61229b6', 'f5723deba', '1ced7f0b4', 'b9a4f06cd', '8132d18b8', 'df28ac53d', 'ae825156f', '936dc3bc4', '5b233cf72', '95a2e29fc', '882a3da34', '2cb4d123e', '0e1921717', 'c83d6b24d', '90a2428a5', '67e6c62b9', '320931ca8', '900045349', 'bf89fac56', 'da3b0b5bb', 'f06078487', '56896bb36', 'a79522786', '71c2f04c9', '1af96abeb', '4b1a994cc', 'dee843499', '645b47cde', 'a8e15505d', 'cc9c2fc87'],
    ['b6daeae32', '3bdee45be', '3d6d38290', '5a1589f1a', '961b91fe7', '29c059dd2', 'cfc1ce276', '0a953f97e', '30b3daec2', 'fb5f5836e', 'c7525612c', '6fa35fbba', '72d34a148', 'dcc269cfe', 'bdf773176', '469630e5c', '23db7d793', 'dc10234ae', '5ac278422', '6cf7866c1', 'a39758dae', '45f6d00da', '251d1aa17', '84d9d1228', 'b98f3e0d7', '66146c12d', 'd6470c4ce', '3f4a39818', 'f16a196c6', 'b8f892930', '6f88afe65', 'ed8951a75', '371da7669', '4b9540ab3', '230a025ca', 'f8cd9ae02', 'de4e75360', '540cc3cd1', '7623d805a', 'c2dae3a5a'],
    ['d0d340214', '34d3715d5', '9c404d218', 'c624e6627', 'a1b169a3a', 'c144a70b1', 'b36a21d49', 'dfcf7c0fa', 'c63b4a070', '43ebb15de', '1f2a670dd', '3f07a4581', '0b1560062', 'e9f588de5', '65d14abf0', '9ed0e6ddb', '0b790ba3a', '9e89978e3', 'ee6264d2b', 'c86c0565e', '4de164057', '87ba924b1', '4d05e2995', '2c0babb55', 'e9375ad86', '8988e8da5', '8a1b76aaf', '724b993fd', '654dd8a3b', 'f423cf205', '3b54cc2cf', 'e04141e42', 'cacc1edae', '314396b31', '2c339d4f2', '3f8614071', '16d1d6204', '80b6e9a8b', 'a84cbdab5', '1a6d13c4a'],
    ['a9819bda9', 'ea26c7fe6', '3a89d003b', '1029d9146', '759c9e85d', '1f71b76c1', '854e37761', '56cb93fd8', '946d16369', '33e4f9a0e', '5a6a1ec1a', '4c835bd02', 'b3abb64d2', 'fe0dd1a15', 'de63b3487', 'c059f2574', 'e36687647', 'd58172aef', 'd746efbfe', 'ccf6632e6', 'f1c272f04', 'da7f4b066', '3a7771f56', '5807de036', 'b22eb2036', 'b77c707ef', 'e4e9c8cc6', 'ff3b49c1d', '800f38b6b', '9a1d8054b', '0c9b00a91', 'fe28836c3', '1f8415d03', '6a542a40a', 'd53d64307', 'e700276a2', 'bb6f50464', '988518e2d', 'f0eb7b98f', 'd7447b2c5'],
    ['87ffda550', '63c094ba4', '2e103d632', '1c71183bb', 'd5fa73ead', 'e078302ef', 'a6b6bc34a', 'f6eba969e', '0d51722ca', 'ce3d7595b', '6c5c8869c', 'dfd179071', '122c135ed', 'b4cfe861f', 'b7c931383', '44d5b820f', '4bcf15776', '51d4053c7', '1fe5d56b9', 'ea772e115', 'ad009c8b9', '68a945b18', '62fb56487', 'c10f31664', 'cbb673163', 'c8d582dd2', '8781e4b91', 'bd6da0cca', 'ca2b906e8', '11e12dbe8', 'bb0ce54e9', 'c0d2348b7', '77deffdf0', 'f97d9431e', 'a09a238d0', '935ca66a9', '9de83dc23', '861076e21', 'f02ecb19c', '166008929'],
    ['f3cf9341c', 'fa11da6df', 'd47c58fe2', '0d5215715', '555f18bd3', '134ac90df', '716e7d74d', 'c00611668', '1bf8c2597', '1f6b2bafa', '174edf08a', 'f1851d155', '5bc7ab64f', 'a61aa00b0', 'b2e82c050', '26417dec4', '53a550111', '51707c671', 'e8d9394a0', 'cbbc9c431', '6b119d8ce', 'f296082ec', 'be2e15279', '698d05d29', '38e6f8d32', '93ca30057', '7af000ac2', '1fd0a1f2a', '41bc25fef', '0df1d7b9a', '88d29cfaf', '2b2b5187e', 'bf59c51c3', 'cfe749e26', 'ad207f7bb', '11114a47a', '341daa7d1', 'a8dd5cea5', '7b672b310', 'b88e5de84'],
]

NUM_FEATURES = 40
for group in colgroups:
    assert len(group) == NUM_FEATURES

giba_cols = colgroups[0]
colgroups_flattened = sum(colgroups, [])
train = pd.read_csv('../input/train.csv').drop(columns=['ID'])
train.drop(train.columns[train.nunique() < 10], axis=1, inplace=True)
def build_histograms(df):
    df_X = (df.replace(0, np.nan).apply(np.log) * 10).round()
    start = int(df_X.min().min())
    stop = int(df_X.max().max())
    return pd.DataFrame(data={'bucket{}'.format(cnt): (df_X == cnt).sum() for cnt in trange(start, stop + 1)})

df = build_histograms(train)
df.head(5)
tsne_res = TSNE(n_components=3, perplexity=40, 
                verbose = 2, early_exaggeration=60, 
                learning_rate=150).fit_transform(df)
tsne_scaled = StandardScaler().fit_transform(tsne_res)

from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure(figsize=(7, 7))
ax = Axes3D(fig)
plt.title("TSNE Result")

#create masks to separate plotted groups by color
colgroups_mask = np.array([f in colgroups_flattened for f in df.index])    
target_idx = np.where(df.index == "target")

ax.scatter(tsne_scaled[~colgroups_mask, 0], tsne_scaled[~colgroups_mask, 1], tsne_scaled[~colgroups_mask, 2],
            s=6, alpha=1, c="C0", label = "Other")
ax.scatter(tsne_scaled[colgroups_mask, 0], tsne_scaled[colgroups_mask, 1], tsne_scaled[colgroups_mask, 2],
           s=8, alpha=1, c="C1", label = "Existing groups")
ax.scatter(tsne_scaled[target_idx, 0], tsne_scaled[target_idx, 1], tsne_scaled[target_idx, 1], 
           s=20, alpha=1, c="C2", label = "\"target\"")

r = [-2.5, 2.5]
ax.set_xlim(r)
ax.set_ylim(r)
ax.set_zlim(r)

plt.legend()
plt.show()
dbscan = DBSCAN(eps=0.2, min_samples=15).fit(tsne_scaled)
print("DBSCAN found {} clusters".format(len(np.unique(dbscan.labels_))))
fig = plt.figure(figsize=(7, 7))
ax = Axes3D(fig)
plt.title("DBSCAN clusters")

from matplotlib.colors import hsv_to_rgb
unique = np.unique(dbscan.labels_)
colorscheme = [hsv_to_rgb((1.618*i % 1, 1, 1)) for i in range(len(unique) + 1)] #generate distinct colors
colorscheme[0] = [0, 0, 0] #noise points should be black

for label in tqdm(unique):
    label_points = tsne_scaled[dbscan.labels_ == label]
    
    #scatter label points
    color = colorscheme[label + 1]
    ax.scatter(label_points[:, 0], label_points[:, 1], label_points[:, 2],
               s=8, alpha=0.35)
    
    #draw label
    text = str(len(label_points))
    ax.text(np.mean(label_points[:, 0]), np.mean(label_points[:, 1]), np.mean(label_points[:, 2]),
            text)
    
ax.set_xlim(r)
ax.set_ylim(r)
ax.set_zlim(r)

plt.show()
    
print("{}% of the data was identified as noise by DBSCAN".format(np.mean(dbscan.labels_ == -1)))
unique, counts = np.unique(dbscan.labels_, return_counts=True)
print("{} groups already had length {}".format(np.sum(counts == NUM_FEATURES), NUM_FEATURES))

extracted = [] #list to hold extracted feature name groups
correct_cluster_labels = unique[counts == NUM_FEATURES]
for label in correct_cluster_labels:
    fnames = df.index[np.where(dbscan.labels_ == label)]
    extracted.append(fnames.tolist())   
#Search for subgroups of larger groups
print("{} groups had length an exact multiple of {}\n".format(
    np.sum((counts != NUM_FEATURES) & (counts % NUM_FEATURES == 0)), NUM_FEATURES))

sub_extracted = []
for label in unique[(counts != NUM_FEATURES) & (counts % NUM_FEATURES == 0)]:
    filtered_data = tsne_scaled[dbscan.labels_ == label, :]
    filtered_indexes = df.index[dbscan.labels_ == label]
    #cluster with smaller epsilon
    sub_dbscan = DBSCAN(eps=0.15, min_samples=15).fit(filtered_data)
    sub_uniques, sub_counts = np.unique(sub_dbscan.labels_, return_counts=True)
    for found_label in sub_uniques[sub_counts == NUM_FEATURES]:
            fnames = filtered_indexes[sub_dbscan.labels_ == found_label]
            sub_extracted.append(fnames.tolist())       
        
            
print("{} groups were successfully extracted by splitting".format(len(sub_extracted)))

#Visualize groups found
fig = plt.figure(figsize=(7, 7))
ax = Axes3D(fig)
plt.title("Extracted groupings visualization")

def plot_groups(groups, color, label):
    group_flattened = sum(groups, [])
    filtered = tsne_scaled[[f in group_flattened for f in df.index], :]    
    ax.scatter(filtered[:, 0], filtered[:, 1], filtered[:, 2], 
                zorder=1, s=10, c=color)

plot_groups(extracted, "C1", "Direct extracted")
plot_groups(sub_extracted, "C2", "Subgroup extracted")

plt.legend()
plt.show()
extracted_colgroups = extracted + sub_extracted
print(len(extracted_colgroups))
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

def do_order_color_plot(groups):
    
    fig = plt.figure(figsize=(7,7))
    ax = Axes3D(fig)
    cmap = ScalarMappable(cmap=plt.get_cmap('winter'), norm=Normalize(vmin=0, vmax=NUM_FEATURES))
    
    for i in range(NUM_FEATURES):
        for colgroup in groups:
            fname = colgroup[i]
            p = tsne_scaled[df.index == fname, :]
            ax.scatter(p[0, 0], p[0, 1], p[0, 2], c = cmap.to_rgba(i))

do_order_color_plot(colgroups)
plt.title("Groups color coded by timeseries order")
plt.show()
do_order_color_plot(extracted_colgroups)
from sklearn.decomposition import PCA

pca = PCA(n_components=1)

for i in trange(len(extracted_colgroups)):
    group = extracted_colgroups[i]
    filtered = tsne_scaled[[f in group for f in df.index], :]    
    
    pca_res = pca.fit_transform(filtered)
    key_func = lambda x: -pca_res[group.index(x)]
    
    group = sorted(group, key=key_func)
    extracted_colgroups[i] = group
    
do_order_color_plot(extracted_colgroups)
fig = plt.figure(figsize=(7,7))
ax = Axes3D(fig)
plt.title("Order Validation Errors")

cmap = ScalarMappable(cmap=plt.get_cmap('winter'), norm=Normalize(vmin=0, vmax=NUM_FEATURES))
    
incorrect = [0] * len(colgroups)
for i in range(NUM_FEATURES):    
    for group in extracted_colgroups:
        for known_group in colgroups:
            if group[0] in known_group:
                match = (group[i] == known_group[i])
                p = tsne_scaled[df.index == group[i], :]
                
                if match:                    
                    ax.scatter(p[0, 0], p[0, 1], p[0, 2], 
                           alpha = 0.3, c = cmap.to_rgba(i))
                else:
                    incorrect[colgroups.index(known_group)] += 1
                    ax.scatter(p[0, 0], p[0, 1], p[0, 2], 
                           alpha = 1, c="r")
                
print(incorrect)
fig = plt.figure(figsize=(7,7))
ax = Axes3D(fig)
cmap = ScalarMappable(cmap=plt.get_cmap('winter'), norm=Normalize(vmin=0, vmax=NUM_FEATURES))

for i in range(NUM_FEATURES):
    for colgroup in extracted_colgroups:
        fname = colgroup[i]
        p = tsne_scaled[df.index == fname, :]
        ax.scatter(p[0, 0], p[0, 1], p[0, 2], c = cmap.to_rgba(i))

for i, group in enumerate(extracted_colgroups):
    p = tsne_scaled[df.index == group[NUM_FEATURES/2], :]
    ax.text(p[0, 0], p[0, 1], p[0, 2], str(i), zorder=2)

#remove colgroups from extracted_colgroups
for group in extracted_colgroups:
    if group[0] in colgroups_flattened:
        extracted_colgroups.remove(group)

#re add the colgroups to the front in their original order
extracted_colgroups = colgroups + extracted_colgroups
extracted_df = pd.DataFrame(extracted_colgroups)
extracted_df.to_csv("groups.csv", header=False, index=False)