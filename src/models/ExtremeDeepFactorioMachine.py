#                                 ___..........__
#           _,...._           _."'_,.++8n.n8898n.`"._        _....._
#         .'       `".     _.'_.'" _.98n.68n. `"88n. `'.   ,"       `.
#        /        .   `. ,'. "  -'" __.68`""'""=._`+8.  `.'     .     `.
#       .       `   .   `.   ,d86+889" 8"""+898n, j8 9 ,"    .          \
#      :     '       .,   ,d"'"   _..d88b..__ `"868' .'  . '            :
#      :     .      .    _    ,n8""88":8"888."8.  "               '     :
#       \     , '  , . .88" ,8P'     ,d8. _   `"8n  `+.      `.   .     '
#        `.  .. .     d89' "  _..n689+^'8n88n.._ `+  . `  .  , '      ,'
#          `.  . , '  8'    .d88+"    j:""' `886n.    b`.  ' .' .   ."
#           '       , .j            ,d'8.         `  ."8.`.   `.  ':
#            .    .' n8    ,_      .f A 6.      ,..    `8b, '.   .'_
#          .' _    ,88'   :8"8    6'.d`i.`b.   d8"8     688.  ".    `'
#        ," .88  .d868  _         ,9:' `8.`8   "'  ` _  8+""      b   `,
#      _.  d8P  d'  .d :88.     .8'`j   ;+. "     n888b 8  .     ,88.   .
#     `   :68' ,8   88     `.   '   :   l `     .'   `" jb  .`   688b.   ',
#    .'  .688  6P   98  =+""`.      `   '       ,-"`+"'+88b 'b.  8689  `   '
#   ;  .'"888 .8;  ."+b. : `" ;               .: "' ; ,n  `8 q8, '88:       \
#   .   . 898  8:  :    `.`--"8.              d8`--' '   .d'  ;8  898        '
#  ,      689  9:  8._       ,68 .        .  :89    ..n88+'   89  689,' `     .
#  :     ,88'  88  `+88n  -   . .           .        " _.     6:  `868     '   '
#  , '  .68h.  68      `"    . . .        .  . .             ,8'   8P'      .   .
#  .      '88  'q.    _.f       .          .  .    '  .._,. .8"   .889        ,
# .'     `898   _8hnd8p'  ,  . ..           . .    ._   `89,8P    j"'  _   `
#  \  `   .88, `q9868' ,9      ..           . .  .   8n .8 d8'   +'   n8. ,  '
#  ,'    ,+"88n  `"8 .8'     . ..           . .       `8688P"   9'  ,d868'   .  .
#  .      . `86b.    " .       .            ..          68'      _.698689;      :
#   . '     ,889_.n8. ,  ` .   .___      ___.     .n"  `86n8b._  `8988'b      .,6
#    '       q8689'`68.   . `  `:. `.__,' .:'  ,   +   +88 `"688n  `q8 q8.     88
#    , .   '  "     `+8 n    .   `:.    .;'   . '    . ,89           "  `q,    `8
#   .   .   ,        .    + c  ,   `:.,:"        , "   d8'               d8.    :
#    . '  8n           ` , .         ::    . ' "  .  .68h.             .8'`8`.  6
#     ,    8b.__. ,  .n8688b., .    .;:._     .___nn898868n.         n868b "`   8
#      `.  `6889868n8898886888688898"' "+89n88898868868889'         688898b    .8
#       :    q68   `""+8688898P ` " ' . ` '  ' `+688988P"          d8+8P'  `. .d8
#       ,     88b.       `+88.     `   ` '     .889"'           ,.88'        .,88
#        :    '988b        "88b.._  ,_      . n8p'           .d8"'      '     689
#        '.     "888n._,      `"8"+88888n.8,88:`8 .     _ .n88P'   .  `      ;88'
#         :8.     "q888.  .            "+888P"  "+888n,8n8'"      .  .     ,d986
#         :.`8:     `88986                          `q8"           ,      :688"
#         ;.  '8,      "88b .d                        '                  ,889'
#         :..   `6n      '8988                                         b.89p
#         :. .    '8.      `88b                                        988'
#         :. .      8b       `q8.        '                     . '   .d89      '
#         . .        `8:       `86n,.       " . ,        , "        ,98P      ,
#         .. .         '6n.       +86b.        .      .         _,.n88'     .
#           .            `"8b.      'q98n.        ,     .  _..n868688'          .
#          ' . .            `"98.     `8868.       .  _.n688868898p"            d
#           . .                '88.      "688.       q89888688868"            ,86
#         mh '. .                 88.     `8898        " .889"'              .988
# Art by Maija Haavisto
# Acquired from : https://github.com/rixwew/pytorch-fm
import numpy as np
import torch
import torch.nn.functional as F


class ExtremeDeepFactorizationMachineModel(torch.nn.Module):
    """
    A pytorch implementation of xDeepFM.
    Reference:
        J Lian, et al. xDeepFM: Combining Explicit and Implicit Feature Interactions for Recommender Systems, 2018.
    """

    def __init__(self, field_dims, embed_dim, mlp_dims, dropout, cross_layer_sizes, split_half=True):
        super().__init__()
        self.embedding = FeaturesEmbedding(field_dims, embed_dim)
        self.embed_output_dim = len(field_dims) * embed_dim
        self.cin = CompressedInteractionNetwork(len(field_dims), cross_layer_sizes, split_half)
        self.mlp = MultiLayerPerceptron(self.embed_output_dim, mlp_dims, dropout)
        self.linear = FeaturesLinear(field_dims)

    def forward(self, x):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        """
        embed_x = self.embedding(x)
        x = self.linear(x) + self.cin(embed_x) + self.mlp(embed_x.view(-1, self.embed_output_dim))
        #x = torch.clamp(x, min=0.8, max=5.2)

        return x.squeeze(1)


class CompressedInteractionNetwork(torch.nn.Module):

    def __init__(self, input_dim, cross_layer_sizes, split_half=True):
        super().__init__()
        self.num_layers = len(cross_layer_sizes)
        self.split_half = split_half
        self.conv_layers = torch.nn.ModuleList()
        prev_dim, fc_input_dim = input_dim, 0
        for i in range(self.num_layers):
            cross_layer_size = cross_layer_sizes[i]
            self.conv_layers.append(torch.nn.Conv1d(input_dim * prev_dim, cross_layer_size, 1,
                                                    stride=1, dilation=1, bias=True))
            if self.split_half and i != self.num_layers - 1:
                cross_layer_size //= 2
            prev_dim = cross_layer_size
            fc_input_dim += prev_dim
        self.fc = torch.nn.Linear(fc_input_dim, 1)

    def forward(self, x):
        """
        :param x: Float tensor of size ``(batch_size, num_fields, embed_dim)``
        """
        xs = list()
        x0, h = x.unsqueeze(2), x
        for i in range(self.num_layers):
            x = x0 * h.unsqueeze(1)
            batch_size, f0_dim, fin_dim, embed_dim = x.shape
            x = x.view(batch_size, f0_dim * fin_dim, embed_dim)
            x = F.relu(self.conv_layers[i](x))
            if self.split_half and i != self.num_layers - 1:
                x, h = torch.split(x, x.shape[1] // 2, dim=1)
            else:
                h = x
            xs.append(x)
        return self.fc(torch.sum(torch.cat(xs, dim=1), 2))


class FeaturesEmbedding(torch.nn.Module):

    def __init__(self, field_dims, embed_dim):
        super().__init__()
        self.embedding = torch.nn.Embedding(sum(field_dims), embed_dim)
        self.offsets = np.array((0, *np.cumsum(field_dims)[:-1]), dtype=np.long)
        torch.nn.init.xavier_uniform_(self.embedding.weight.data)

    def forward(self, x):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        """
        x = x + x.new_tensor(self.offsets).unsqueeze(0)
        return self.embedding(x)


class FeaturesLinear(torch.nn.Module):

    def __init__(self, field_dims, output_dim=1):
        super().__init__()
        self.fc = torch.nn.Embedding(sum(field_dims), output_dim)
        self.bias = torch.nn.Parameter(torch.zeros((output_dim,)))
        self.offsets = np.array((0, *np.cumsum(field_dims)[:-1]), dtype=np.long)

    def forward(self, x):
        """
        :param x: Long tensor of size ``(batch_size, num_fields)``
        """
        x = x + x.new_tensor(self.offsets).unsqueeze(0)
        return torch.sum(self.fc(x), dim=1) + self.bias



class MultiLayerPerceptron(torch.nn.Module):

    def __init__(self, input_dim, embed_dims, dropout, output_layer=True):
        super().__init__()
        layers = list()
        for embed_dim in embed_dims:
            layers.append(torch.nn.Linear(input_dim, embed_dim))
            layers.append(torch.nn.BatchNorm1d(embed_dim))
            layers.append(torch.nn.ReLU())
            layers.append(torch.nn.Dropout(p=dropout))
            input_dim = embed_dim
        if output_layer:
            layers.append(torch.nn.Linear(input_dim, 1))
        self.mlp = torch.nn.Sequential(*layers)

    def forward(self, x):
        """
        :param x: Float tensor of size ``(batch_size, embed_dim)``
        """
        return self.mlp(x)