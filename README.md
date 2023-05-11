
# bulkGAN

<!-- badges: start -->
<!-- badges: end -->

The goal of bulkGAN is to generate RNA data close to real bulk RNA-seq with the constraints of scRNA-seq data, based on keras and tensflow framework.

## Installation

You can install the development version of bulkGAN like so:

``` r
library(bulkGAN)
devtools::install_git('liufan-creat/bulkGAN')
```

## Example

This is a basic example which shows you how to solve a common problem:

``` r
library(keras)
library(reticulate)
library(tensorflow)
library(bulkGAN)
tfa <- import('tensorflow_addons') 
## data preprocessing code
## The model needs to provide gene expression matrix of bulk RNA-seq and scRNA-seq，They need to have the same row name，which is the co-expression gene and require the same gene order. We recommend providing scRNA data with library normalization, although raw data is also acceptable. Total count of each single cell is needed before filtering.


RNA <- read.csv('~/Bio_project/snRNA/bulk-RNA.csv',
                stringsAsFactors = F,comment.char = '#',sep = '')
single_cell <-  read.csv('~/Bio_project/snRNA/getx.csv', row.names =1)

count <-  apply(single_cell,2,sum)

gene <- Reduce(intersect,list(rownames(exp),rownames(single_cell) ))
## 
normal_tissure <- sn_exp[which(rownames(sn_exp) %in% gene),]
normal_single_cell <- single_cell[which(rownames(single_cell) %in% gene),]

## tpm tissure 
normal_tissure$Gene <- rownames(normal_tissure)  
normal_tissure$Gene <- factor(normal_tissure$Gene,levels = rownames(normal_single_cell))
normal_tissure <- normal_tissure[order(normal_tissure$Gene  ),]
normal_tissure <- normal_tissure[,-which(colnames(normal_tissure) == 'Gene')]


Tsln <- transform_normalization(output,normal_single_cell,custom_count=T,count)

### SNGAN
## output_shape = n.cell, intput_shape = n.gene
generator <- SNGAN_generator(node=c(128,256,512),output_shape= 800L)
generator

discriminator <- SNGAN_discriminator(node=c(256,128,64,16,8),
                                    input_shape= 15400L,dropout_rate = 0.2)
discriminator

fit_SNGAN(generator, discriminator,n.discriminator=5,batch=25,
          epoch = 500,learn_rate = 0.0001,object = normal_tissure )

### pairwise-GAN
generator <- SNGAN_generator(node=c(128,256,512),output_shape= 800L)
generator

discriminator <- pairwise_discriminator(node=c(256,128,64,16,8),
                                        input_shape= 15400L,dropout_rate = 0.2)
discriminator

fit_discriminator(generator, discriminator,n.discriminator=5,batch=25,
                  epoch = 500,learn_rate = 0.0001,object = normal_tissure )

### triplet-cGAN


```

