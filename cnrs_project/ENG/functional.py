import torch
import numpy as np
from tqdm import tqdm
import numpy as np
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
import torch.nn.functional as F
import sys


class Logger(object):
    def __init__(self, filename="Default.log"):
        self.terminal = sys.stdout
        self.log = open(filename, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass

class FocalLoss(torch.nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, num_classes=3):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.num_classes = num_classes

    def forward(self, inputs, targets):
        targets = F.one_hot(targets, self.num_classes).float()
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss
        return F_loss.mean()


def vali(args, accelerator, model, vali_loader, criterion, dic_cat_labels):
    total_loss = []
    predictions, true_labels = [], []
    dic_cat_labels = {value: key for key, value in dic_cat_labels.items()}  # Reverse keys and values

    model.eval()
    with torch.no_grad():
        for i, (batch_date, batch_text, batch_win, batch_lab) in tqdm(enumerate(vali_loader)):
            batch_win = batch_win.float().to(accelerator.device)
            batch_lab = batch_lab.to(torch.int64).to(accelerator.device)

            if args.use_amp:
                with torch.cuda.amp.autocast():
                    if args.output_attention:
                        outputs = model(batch_text, batch_win)[0]
                    else:
                        outputs = model(batch_text, batch_win)

            else:
                if args.output_attention:
                    outputs = model(batch_text, batch_win)[0]
                else:
                    outputs = model(batch_text, batch_win)

            outputs, batch_lab = accelerator.gather_for_metrics((outputs, batch_lab))

            pred = outputs.detach()
            true = batch_lab.detach()

            loss = criterion(pred, true)
            total_loss.append(loss.item())

            predictions.extend(pred.cpu().numpy())
            true_labels.extend(true.cpu().numpy())

    total_loss = np.average(total_loss)

    model.train()

    pred_flat_cat = np.argmax(predictions, axis=1)
    true_labels_cat = [dic_cat_labels.get(x) for x in true_labels]
    pred_flat_cat = [dic_cat_labels.get(x) for x in pred_flat_cat]

    cr = classification_report(true_labels_cat, pred_flat_cat, digits=4,
                               labels=list(dic_cat_labels.values()), zero_division=0)

    return total_loss, cr