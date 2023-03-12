import os

import numpy as np # linear algebra

from dicom import read_file as read_dicom_image

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

from glob import glob

from itertools import groupby

import matplotlib.pyplot as plt

from skimage.color import label2rgb

from skimage.segmentation import mark_boundaries
# get all of the patients

all_images = glob(os.path.join('..', 'input', 'sample_images', '*', '*'))

folder_list = [(k, list(v)) for k, v in groupby(all_images, lambda x: os.path.split(os.path.dirname(x))[1])]
n_img = read_dicom_image(all_images[150])

plt.hist(n_img.pixel_array.flatten(),np.linspace(-500, 500, 20))
# Calculate features for each image

def show_area(in_dcm, min_val, max_val):

    in_img = in_dcm.pixel_array

    lab_img = (in_img>=min_val) 

    lab_img &= (in_img<=max_val)

    in_img = ((in_img+1200.0)/1800.0).clip(0,1)

    return label2rgb(lab_img, image = in_img, bg_label = 0)

def calc_area(in_dcm, min_val, max_val):

    pix_area = np.prod(in_dcm.PixelSpacing)

    return pix_area*np.sum((in_dcm.pixel_array>=min_val) & (in_dcm.pixel_array<=max_val))

feature_list = {

    'blood': (30, 45),

    'fat': (-100, -50),

    'water': (-10, 10)

}
plt.imshow(show_area(n_img, *feature_list['blood']))
has_cancer = ['0015ceb851d7251b8f399e39779d1e7d',

 '006b96310a37b36cccb2ab48d10b49a3',

 '008464bb8521d09a42985dd8add3d0d2',

 '00edff4f51a893d80dae2d42a7f45ad1',

 '0257df465d9e4150adef13303433ff1e',

 '02801e3bbcc6966cb115a962012c35df',

 '028996723faa7840bb57f57e28275e4c',

 '04a8c47583142181728056310759dea1',

 '05609fdb8fa0895ac8a9be373144dac7',

 '059d8c14b2256a2ba4e38ac511700203',

 '064366faa1a83fdcb18b2538f1717290',

 '0708c00f6117ed977bbe1b462b56848c',

 '07349deeea878c723317a1ce42cc7e58',

 '07bca4290a2530091ce1d5f200d9d526',

 '081f4a90f24ac33c14b61b97969b7f81',

 '09d7c4a3e1076dcfcae2b0a563a28364',

 '0acbebb8d463b4b9ca88cf38431aac69',

 '0c0de3749d4fe175b7a5098b060982a1',

 '0c37613214faddf8701ca41e6d43f56e',

 '0c60f4b87afcb3e2dfa65abbbf3ef2f9',

 '0d06d764d3c07572074d468b4cff954f',

 '0f5ab1976a1b1ef1c2eb1d340b0ce9c4',

 '0ff552aa083ecfabaf1cfd65b0a8e674',

 '118be21b7e0c3058b29a524686391c66',

 '11fe5426ef497bc490b9f1465f1fb25e',

 '12e0e2036f61c8a52ee4471bf813c36a',

 '13bb12b3b27d5a7b4b142503a1ae9e73',

 '1427be78bcf4aba96c5054b697be9b5b',

 '149cc798827099f8bdb97cf702027305',

 '14f713c1ef037f6c531cffdff0e5fb2c',

 '15aa585fb2d3018b295df8619f2d1cf7',

 '169b5bde441e8aa3df2766e2e02cda08',

 '184c61740244f4ce8fb985af9bb3d8e8',

 '184fa4ae2b7ae010625d89f10186f1c5',

 '185bc9d9fa3a58fea90778215c69d35b',

 '198d3ff4979a9a89f78ac4b4a0fe0638',

 '1dab3271160e1380c5a70a1e3ba40cb7',

 '1e0f8048728717064645cb758eb89279',

 '1f49f0c1d7feedcae9024d251797407c',

 '1fb4887efd403cd9c0f6970fc8b679b5',

 '229b8b785f880f61d8dad636c3dc2687',

 '2365e0afe6844e955f3d4c23a16dc1a9',

 '243e69389ae5738d3f89386b0efddbcd',

 '245fe0c86269602b0dab44c345b0b412',

 '2488c5b32e837dc848fe6fe4b1bbb7cb',

 '2619ed1e4eca954af4dcbc4436ef8467',

 '274a81c75d244187247789bd71de2b3a',

 '281bb28a077ccfcd40ce4a543a5aea89',

 '28a9b77a9113ce491433d3ea47fa8fc9',

 '28e29fe26140703e5bbe570f982bd112',

 '2969c7ad0e550fee1f4a68fcb3bbb9e5',

 '29d92a1e253cef2c7f34c6db26ce11e3',

 '2a20e4a4e6411f72374fdffebabfc235',

 '2a2300103f80aadbfac57516d9a95365',

 '2b861ff187c8ff2977d988f3d8b08d87',

 '2c06f5c66f3c79515b7712605dea4400',

 '2d5cd7c1ee9a74a1244ddd6b55ad0446',

 '2d977650e6388d2c45825a77e94437a2',

 '2e8bb42ed99b2bd1d9cd3ffaf5129e4c',

 '2ebb1e8f14802c33f0e4215a7545d70d',

 '2ed8eb4430bf40f5405495a5ec22a76d',

 '2f154a687b94f7b59fec7048cbfb5354',

 '2fc3d8ef26fc7aafad44d5034673dd4c',

 '303b4b8425389134997a38b975c205d3',

 '30b8aa7f5688cab5ff0964f34b715c4d',

 '31136e50b7205e9184227f94cdea0090',

 '318bf8045b625b40825552420abfe1ef',

 '31f35f920a472a1c3eacb565fe027923',

 '322bf0acacba9650fa5656b9613c75c8',

 '3252220375d82c3720d36d757bb17345',

 '3285ba0f447f3091c0c7c061b47c2f62',

 '32cda856b7ec759fd3ebaa363c505e88',

 '33dd6666d9f0338929ecce58bb7c4cc3',

 '3457880b1a66030feb8adaed6da805af',

 '348a53f500ada390ddd00cc47d310b2c',

 '34c0760406297a3c8fd5077fb7cd95b0',

 '352c23fe8a3d0640ea531a6bf223732c',

 '35b9a3e9871499893f76c8e6c648562c',

 '375a52b012066845a2eeb5032a92fc6b',

 '380eb569a5750648434cc8ae8da4a0a9',

 '383c27906392e9ce57f6ab5ef1cb6f62',

 '385f1f49b0c20563177c36b7470f1c46',

 '398208da1bcb6a88e11a7314065f13ff',

 '39ebb8121ea6faec0405a4e8db883b55',

 '3a5bbc2f1f5d6d76a48ba5300105d998',

 '3f6431400c2a07a46386dba3929da45d',

 '4001d754871a8da824b8444e32dc6e0f',

 '40c95c9be0bd7c290534ce374c58bec9',

 '437e42695e7ad0a1cb834e1b3e780516',

 '43f2ef8f53e1aa03bfb6378d0c20a8ac',

 '44988c6efa451e8d496188cb30669d44',

 '4521c94debf37a4dc9f3b70366a21640',

 '48e592418247393234dd658f9112c543',

 '4a782bbc2608288a3ed05e511af6f8bb',

 '4af17bcb31669a9eab0b6ef8e22a8dcf',

 '4baa552f3a11782f39e16b345d401fb8',

 '4cc8af2efef2f41bf70684be25276ce5',

 '4cd70a98baca46b116071b32788d3c2d',

 '4d7df08f074b221eec6311c2617a5ba8',

 '4dbda61d574417c7f25d6e9a8f0749a7',

 '4dcd34bd9b10f96453b63d4f55d1fd44',

 '504e447ad62ea9ebb283873e044b5dd2',

 '51bd5c556c77ecdaf489d8dd9f7a05f1',

 '51fbac477a3639f983904fc4d42b8c15',

 '5267ea7baf6332f29163064aecf6e443',

 '54056288ab97cebc4b0ea33c23f47ff6',

 '5572cc61f851b0d10d6fec913ae722b9',

 '55c01868f1d9c37fa3f174dc3c0d44e8',

 '56462ba8833fc842d16be7e311214404',

 '570ea80b0dcc08f3e8751a6f4b2b1cd5',

 '573a661e2d784f9385a3b78c9757ddad',

 '57822feb6186b788c4e1877123428454',

 '5782e6873c666529c6a66421acb043dc',

 '592c2481f17d6a2cecfe7bbb6a27722c',

 '59fc9d939f05bf3023c1387c1c086520',

 '5ade88428e6463fa212d4c287228e8ed',

 '5b412509bc40a3aeb3b5efef1fdfcfc9',

 '5b642ed9150bac6cbd58a46ed8349afc',

 '5c99ab7172afa78312fe73a3c0dd342f',

 '5e0c8cba8eab51076ac0014049d770c1',

 '5fd33ea74e1ad740a201ae9b3c383fc5',

 '608202eb3c368512e55e9e339a203790',

 '608a7028689c6ab3aea5f116007169b2',

 '60b389fb2f7eeb912586d1a3ccc9dbbc',

 '61630ec628631f7fe3980f869e1a4fbe',

 '6171d57221e26d1f15d3c71fe966ab18',

 '624a34fa8fd36847724e749877343847',

 '627499714e279203bd1294290f8fc542',

 '627836151c555187503dfe472fb15001',

 '63b5be42543c98ac5392f1bfbda085bf',

 '648c99653d512edc1d28dd8e7054ceab',

 '64a5a866461a3b6006efb0075e04dffe',

 '65073aadb60e398d8db1806f5ea2a082',

 '6541df84fd779ba6513a530c128f4e9b',

 '65a380c07d416f78e85545eaaa2916a1',

 '662153a685fb4268361bfbaca5e9ca23',

 '668bb968918c63fad7d65581825b1048',

 '66a92d789e440d3dbef3c69d20e20694',

 '676467220abd8e2104417c5213664ef9',

 '678c5ec1360784e0fe797208069e0bbb',

 '6799964c08ad5ce7740defcd3bd037a6',

 '6857c76be618bb0ddced5f4fecc1695f',

 '6969c031ee0c34053faff3aac9dd2da7',

 '6be677ba1631174397b0c1e26a46af30',

 '6cb2908fd789700db727dd96526bc342',

 '6ee742b62985570a1f3a142eb7e49188',

 '6f43af3f636f37b9695b58378f9265cc',

 '6fd3af9174242c1b393fe4ba515e7a26',

 '713d8136c360ad0f37d6e53b61a7891b',

 '71665cc6a7ee85268ca1da69c94bbaeb',

 '721949894f5309ed4975a67419230a3c',

 '72fd04cf3099b148d9ad361efb988866',

 '733205c5d0bbf19f5c761e0c023bf9a0',

 '7395f64fba89c2463a1b13c400adf876',

 '74b1b748971c474a8023f6406c54b18a',

 '75aef267ad112a21c870b7e2893aaf8a',

 '761aeadb65fb84c8d04978a75b2f684c',

 '763ce10dfdd4662f15de3f5931d5534b',

 '77033e4c1591403d1b1255607a20a983',

 '775c5f8043e72b2284b5885254566271',

 '77d6f5203d46073369d7038b2d58e320',

 '7842c108866fccf9b1b56dca68fc355e',

 '78c0a0104c0428e260cbd9e50eb7eea6',

 '7917af5df3fe5577f912c5168c6307e3',

 '799c0026d66479f7447ed0df5955f051',

 '79e0e507b1cd1d0c8107de4fd6b9e444',

 '7bfba4540956c0b2c5b78b3623a4855d',

 '7c2b72f9e0f5649c22902292febdc89f',

 '7d46ce019d79d13ee9ce8f18e010e71a',

 '7fb1c8ffd78ca4b6869044251add36b4',

 '80600d4a5fee7424d689ba7d0906d50f',

 '817a99e1a60bcf4e37c904d73845ca50',

 '820dd342da11af3a062d1647b3736fdd',

 '823b5f08ce145f837066d2e19dab10c1',

 '8298238a27be6111214a9bc711608181',

 '8326bb56a429c54a744484423a9bd9b5',

 '8369f716ca2d51c934e7f6d44cb156e9',

 '839502f9ff68fd778b435255690f3061',

 '84876a50f52476bcc2a63678257ae8b4',

 '84a6c418d57bfc5214639012998356d4',

 '85d59b470b927e825937ea3483571c6d',

 '8601f5424bcf4cd8e7bc3d649e9995a2',

 '868b024d9fa388b7ddab12ec1c06af38',

 '87cdb87db24528fdb8479220a1854b83',

 '87cdf4626079509e5d6d3c3b6c8bfc2e',

 '880980cc7e88c83b0fea84f078b849e3',

 '8815efa67adb15b2f8cfd49ec992f48e',

 '882107a204c302e27628f85522baea49',

 '88523579f4e325351665753e903cfdf5',

 '88ae66cd575c45ec5bb0f1578e2f1c49',

 '8918c484841c5d0a532fe146e9da61bd',

 '893fbc465b9d8a25569659a2bac154ef',

 '898bd4c517fb9cf94c7d06dae56b0136',

 '89bfbba58ee5cd0e346cdd6ffd3fa3a3',

 '8a97ff581c17a49a3ef97144efde8a19',

 '8b6e16b4e1d1400452956578f8eb97c4',

 '8c63c8ebd684911de92509a8a703d567',

 '8e92c4db434da3b8d4e3cafce3f072fb',

 '8ed68f2dbf103a4bc0fd8708d8c1ac93',

 '8ee6f423ff988d10f2bb383df98c1b2e',

 '90409f7fcfec3581033559f8340e48a9',

 '90e3b396e1c1343a514eb5890833d3d8',

 '90e5f4780b2f05136ff5f776a5cbc2af',

 '91d29bc19205f8eb9a63de5b774a5575',

 '92abfd85dd6afb639e9a8b60aaa08262',

 '9397a41c9e819a92eb5c86e0e652d7c1',

 '93a6f37a72f60498986374f57bfc30c4',

 '95a27273c11db8bfb9fc27b1e64de6bd',

 '9660e4a23b8dd7d5056a622ee3568a41',

 '96acca47671874c41de6023942e10c16',

 '9703fd051751879432975535663150da',

 '9a3174ffe867f602ee82c512a01420ee',

 '9b7524785a9bf40f0651deeb3b05b75f',

 '9c779a4e5e56c77131f8e99d5eacb766',

 '9de4a1ebcdf1cfd8566ed1d9b63cbeb7',

 '9e5c2e760b94b8919691d344cfdbac7f',

 '9e922147900b3984c9345bdda573e882',

 '9e98136d07b953c3362e0a132c8810b6',

 '9f52323d216f89d300612cfac0122d8b',

 'a13d6c5f8f86d74e16c10cf9294bca31',

 'a14e41eea93d7667a87d458d5cb28272',

 'a162d204827e4e89a2e5ba81cc53247a',

 'a19a122fe9a790576b57c6bd5cf9ff5c',

 'a2d9e657a673798f9ebdfec1b361f93a',

 'a32e7fdbc0db97e35aabd7c931a582ea',

 'a4dc34f2731b62f60c6c390a39fe11b2',

 'a532c6f9405e6f3a4229ea6e04b0d975',

 'a70fd23bd8d535ffd42259cb91f4c5ca',

 'a76b682e74918492c1f2ca4c13c29885',

 'a784a51caee14229d46777f2a9770a5f',

 'a853993fd839a0ee61f2ca73c4e497a6',

 'a88c585e7d81744eec091a6f0600bd7b',

 'a8e650f8494e894be06c9cee08702aa9',

 'aa2747369e1a0c724bea611ea7e5ffcf',

 'aa55708fcc8bf27b605bcd2fca0dc991',

 'ac00af80df36484660203d5816d697aa',

 'ac3345a5a05655c6bcce7d0b226a0042',

 'ac366a2168a4d04509693b7e5bcf3cce',

 'ac4c6d832509d4cee3c7ac93a9227075',

 'ac57a379cfea05c07d9befe8b9359495',

 'ac68eb0a3db3de247c26909db4c10569',

 'ac9c16f3f287f0e0b321fb518ac71c75',

 'ad7e6fe9d036ed070df718f95b212a10',

 'adc3bbc63d40f8761c59be10f1e504c3',

 'aea6f1621333074412b9a6acdcda31a9',

 'af4dfdda000c16c4cb77ea236cf1e524',

 'af6d573b8c6804e14e3a7b07a376e593',

 'afb37b10bd304fa2c7b70cfaf1f489ed',

 'b022a1d30d62ef2c1f0902f1a047a845',

 'b158f44c31f4121c865c828ff79fc73d',

 'b17cb533d71d63d548ce47b48b34c23c',

 'b5de57869d863bdc1b84b0194e79a9d3',

 'b635cda3e75b4b7238c18c6a5f1858f6',

 'b6578699374a9954b9a8a9e7da2603b1',

 'b6687898fe385b68d5ae341419ef3fdd',

 'b6d8dd834f2ff1ed7a5154e658460699',

 'b7045ebff6dbb0023087e0399d00b873',

 'b7ef0e864365220b8c8bfb153012d09a',

 'b83ce5267f3fd41c7029b4e56724cd08',

 'b84c43bed6c51182d7536619b747343a',

 'b8bb02d229361a623a4dc57aa0e5c485',

 'b8dc33b670bb078d10954345c3ffbb3a',

 'ba71b330a16e8b4c852f9a8730ee33b9',

 'bb4b43d0dc4d9d2b61150df6556f6490',

 'bbe21f027a1df4b07016b474b48d3f65',

 'bc38f78d1194f57452f6bb5eed453137',

 'bda661b08ad77afb79bd35664861cd62',

 'be2be08151ef4d3aebd3ea4fcd5d364b',

 'c020f5c28fc03aed3c125714f1c3cf2a',

 'c05acf3bd59f5570783138c01b737c3d',

 'c0625c79ef5b37e293b5753d20b00c89',

 'c0f0eb84e70b19544943bed0ea6bd374',

 'c1673993c070080c1d65aca6799c66f8',

 'c1ba619e3b49e0cb7798bd10465c2b29',

 'c2bdfb6ab5192656b397459648221918',

 'c2e546795f1ea2bd7f89ab3b4d13e761',

 'c3b05094939cc128a4593736f05eadec',

 'c3e8db4f544e2d4ecb01c59551eb8ef0',

 'c4c801ae039ba335fa32df7d84e4decb',

 'c5887c21bafb90eb8534e1a632ff2754',

 'c610439ebef643c7fd4b30de8088bb55',

 'c67de8fbbe1e58b464334f93a1dd0447',

 'c67e799bcc1e2635eb9164f6e8cf75f3',

 'c8cfb917b0d619cb4e25f789db4641f8',

 'cb64ff663195832e0b66a9bb17891954',

 'cb94c3f894fc93c1ec0eb436c8564ed3',

 'cd104ad99d5b939b6bdd28b154e28085',

 'cd10ceca9862ba0cc2ffd0ed8c9b055c',

 'cf0a772e90a14d77d664ce9baedf0e5c',

 'cfbcb16fea277226d6771d8b1966397a',

 'd09e4124b97b22ef45692b62b4ca7f03',

 'd244870d213a21efa86e86c951d8c9a2',

 'd2b47d9034d38a410f00dabba9754d91',

 'd2eecd9f13a6d474338045d0c91cffbd',

 'd6d5ed3055d084a6abf0f97af3fe2ff0',

 'd7713d80767cfdaad5e3db537849d8d0',

 'd777a77cc7a2ec2f1eed68799cc9075c',

 'd7aa27d839b1ecb03dbf011af4bcb092',

 'd7e5640b52c8e092ec277febc81478da',

 'd81704ee56c124cc1434640918a3299c',

 'd8ed783494996f55a587270a212f7d5b',

 'd917c781760710015473eee9ce82e051',

 'd92998a73d4654a442e6d6ba15bbb827',

 'd991b1760fb8705de655a1da068f7a6a',

 'd9a2bea7df4a888313374beb142cf9c0',

 'da821546432756d377777d7f4c41ca2f',

 'dbfbc12c7a943a2dc0e34bfd4a636bca',

 'dc9854bcdcc71b690d9806438009001d',

 'df015da931ad5312ee7b24b201b67478',

 'df761dd787bfc439890740ccce934f36',

 'e00832e96709eb85f8e0e608ca02c2b5',

 'e10c2b829c39d4a500c09caf04d461a1',

 'e2b7fe7fbb002029640c0e65e3051888',

 'e3423505ef6b43f03c5d7bde52a5a78c',

 'e38789c5eabb3005bfb82a5298055ba0',

 'e43afa905c8e279f818b2d5104f6762b',

 'e537c91cdfa97d20a39df7ef04a52570',

 'e54b574a7e7c650edc224cbdede9e675',

 'e56b9f25a47a42f4ae4085005c46109c',

 'e572e978c2b50aca781e6302937e5b13',

 'e58b78dc31d80a50285816f4ecd661e3',

 'e5c68cfa0f33540da3098800f0daae2c',

 'e5cf847e616cc2fe94816ffa547d2614',

 'e659f6517c4df17e86d4d87181396ea6',

 'e6b3e750c6c7a70ca512d77defcfe615',

 'e709901da9ba15a95d4a29906edc01dd',

 'e8eb842ee04bbad407f85fe671f24d4f',

 'e9ccf1ce85c39779fafb9ec703c71555',

 'ea7373271a2441b5864df2053c0f5c3e',

 'eaf753dc137e12fd06e96d27f3111043',

 'eb008af181f3791fdce2376cf4773733',

 'ebd601d40a18634b100c92e7db39f585',

 'ed0f3c1619b2becec76ba5df66e1ea56',

 'ed49b57854f5580658fb3510676e03dd',

 'eed4db0cb0576c274de569e98a56a270',

 'f17867cc3e579dc2fc6f0334bc43a91d',

 'f1a64fda219db48bcfb8ad3823ef9fc1',

 'f25c425c827b35fcbaa23f2ed671540b',

 'f29d00ddf6d9846aa600c3f0edf5f952',

 'f2ca85bb9ae82a3d79b9f321f727ac19',

 'f2f23b265b2a3b977cb81fe3193d7c2c',

 'f3eafe72b1e9528116f3c430ab73a2ae',

 'f3f6f40ccb01276d722d52701cab1754',

 'f42a0343e5b5154c6a184fc955d8f20f',

 'f467795ce3b50a771085d79ae8d29ecc',

 'f5717f7cbc08d8bd942cd4c1128e3339',

 'f63f2f63e2619012b4c798fd638c8b8a',

 'f6c9e875d7adfe7add08f43528810f72',

 'f725f46908f16062fd12c141eb47c6a7',

 'f76143416ee2c8e1251f45f108fed468',

 'f7a03adba817f2a2249b9dee0586f4be',

 'f8f66fca04d2e67eacd86ea154827a4c',

 'f938f9022abf7f1072fe9df79db7eccd',

 'fa45178d023325b255a3d4fc3e96cb7d',

 'fb57fc6377fd37bb5d42756c2736586c',

 'fb99a80cbb2f441bb90135bab5b029fe',

 'fbae4d04285789dfa32124c86586dd09',

 'fc545aa2f58509dc6d81ef02130b6906',

 'fd0c2dfe0b0c58330675c3191cef0d5b',

 'fd2dd970bd3d91e5b26d7e57c03f70af',

 'fda187bfb1d6a2ecd4abd862c7f7f94c',

 'fe45462987bacc32dbc7126119999392']
simple_features = [{'patient': c_folder, 'cancer': c_folder in has_cancer, 

                    'slice_count':len(c_files)} for c_folder, c_files in folder_list]
simple_df = pd.DataFrame(simple_features)

simple_df.sample(3)
simple_df.plot.scatter('slice_count', 'cancer')
blood_features = [{'patient': c_folder, 'cancer': c_folder in has_cancer, 

                    'blood_in_first_slice': calc_area(read_dicom_image(list(c_files)[0]), *feature_list['blood'])} for c_folder, c_files in folder_list]

bf_df = pd.DataFrame(blood_features)

bf_df.sample(5)
bf_df.plot.scatter('blood_in_first_slice', 'cancer')
bf_df.plot.hist('blood_in_first_slice')
# needed imports for hierarchical clustering

from matplotlib import pyplot as plt

from scipy.cluster.hierarchy import dendrogram, linkage

import numpy as np
# Getting water feature data

water_and_blood_features = [{'patient': c_folder, 'cancer': c_folder in has_cancer, 

                   'water_in_first_slice': calc_area(read_dicom_image(list(c_files)[0]), *feature_list['water']),

                   'blood_in_first_slice': calc_area(read_dicom_image(list(c_files)[0]), *feature_list['blood'])} for c_folder, c_files in folder_list]

features_df = pd.DataFrame(water_and_blood_features)

features_df.sample(5)
# a scatter plot  of blood and water

features_np = features_df.as_matrix(['water_in_first_slice', 'blood_in_first_slice'])

print(features_np)


plt.scatter(features_np[:,0], features_np[:,1])
# Generate the linkage matrix

Z = linkage(features_np, 'ward')
from scipy.cluster.hierarchy import cophenet

from scipy.spatial.distance import pdist



c, coph_dists = cophenet(Z, pdist(features_np))

c
# calculate full dendrogram

plt.figure(figsize=(25, 10))

plt.title('Hierarchical Clustering Dendrogram')

plt.xlabel('sample index')

plt.ylabel('distance')

dendrogram(

    Z,

    leaf_rotation=90.,  # rotates the x axis labels

    leaf_font_size=8.,  # font size for the x axis labels

)

plt.show()
from scipy.cluster.hierarchy import fcluster

max_d = 25000

clusters = fcluster(Z, max_d, criterion='distance')

clusters
plt.figure(figsize=(10, 8))

plt.scatter(features_np[:,0], features_np[:,1], c=clusters, cmap='prism')  # plot points with cluster dependent colors

plt.show()