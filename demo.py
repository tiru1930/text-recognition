import torch
from torch.autograd import Variable
import utils
import dataset
from PIL import Image
import models.crnn as c_rnn


def getPrediction(imagePath):
    model_path = './expr/netCRNN_999.pth'
    alphabet = '0123456789abcdefghijklmnopqrstuvwxyz'
    model = c_rnn.CRNN(32, 1, 37, 256)
    if torch.cuda.is_available():
        # model = model.cuda()
        model = torch.nn.DataParallel(model).cuda()
    print('loading pretrained model from %s' % model_path)
    model.load_state_dict(torch.load(model_path))
    converter = utils.strLabelConverter(alphabet)
    transformer = dataset.resizeNormalize((100, 32))
    image = Image.open(imagePath).convert('L')
    image = transformer(image)
    if torch.cuda.is_available():
        image = image.cuda()
    image = image.view(1, *image.size())
    image = Variable(image)
    print(image.size())
    model.eval()
    preds = model(image)
    print(preds)
    _, preds = preds.max(2)
    preds = preds.transpose(1, 0).contiguous().view(-1)

    preds_size = Variable(torch.IntTensor([preds.size(0)]))
    raw_pred = converter.decode(preds.data, preds_size.data, raw=True)
    sim_pred = converter.decode(preds.data, preds_size.data, raw=False)
    print('%-20s => %-20s' % (raw_pred, sim_pred))


getPrediction("./data/IIIT5K/test/1_1.png")