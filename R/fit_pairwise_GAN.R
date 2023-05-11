


#' Title Fits the model on data yielded batch-by-batch by pairwise-GAN
#'
#' @param generator A generator (e.g. like the one provided by flow_images_from_directory() or a custom R generator function).
#' The output of the generator must be a list of one of these forms: <br> - (inputs, targets) <br> - (inputs, targets, sample_weights) <br>.
#'  This list (a single output of the generator) makes a single batch.
#'  Therefore, all arrays in this list must have the same length (equal to the size of this batch).
#'  Different batches may have different sizes. For example, the last batch of the epoch is commonly smaller than the others, if the size of the dataset is not divisible by the batch size.
#'  The generator is expected to loop over its data indefinitely. An epoch finishes when steps_per_epoch batches have been seen by the model.
#' @param discriminator network for authenticity
#' @param n.discriminator Integer.Number of cycle for discriminator in each epoch
#' @param batch Integer.Number of samples in each training
#' @param epoch Integer. Number of epochs to train the model
#' An epoch is an iteration over the entire data provided,Note that in conjunction with initial_epoch,epochs is to be understood as “final epoch”.
#' The model is not trained for a number of iterations given by epochs,  but merely until the epoch of index epochs is reached.
#' @param learn_rate 	float >= 0. Learning rate.
#' @param object RNAseq matrix
#'
#' @return epoch, gloss, dloss, pearson correlations
#' @export
#'
#' @examples
#' \dontrun{
#' library(keras)
#' library(reticulate)
#' library(tensorflow)
#' library(bulkGAN)
#' library(Hmisc)
#' tfa <- import('tensorflow_addons')  #   import tensorflow_addons model
#'
#' fit_pairwise_GAN(generator, discriminator,n.discriminator=5,batch=25,
#'            epoch = 500,learn_rate = 0.0001,object = normal_tissure )
#'
#'}

fit_pairwise_GAN <- function(generator, discriminator,batch=25,n.discriminator=5,
                             epoch = 500,learn_rate = 0.0001,object = NULL ){


  wloss <- function(y_true,y_pred){
    loss <- -1 *  keras::k_mean(y_true * y_pred)
    return(loss)
  }

  l <- 100

  dloss <- NULL
  gloss <- NULL

  g_coor_p <- NULL


  discriminator %>% keras::compile(optimizer = keras::optimizer_rmsprop(learning_rate = learn_rate),
                            loss = wloss )

  freeze_weights(discriminator)

  input <- keras::layer_input(shape = l)
  neg <- generator(input)
  input_neg <- keras::layer_input(shape = shape)

  validity <- discriminator(list(neg,input_neg))

  GAN <- keras::keras_model(list(input,input_neg),validity)

  GAN %>% keras::compile(optimizer =  keras::optimizer_rmsprop(learning_rate = learn_rate ),
                  loss = wloss)


  for (i in 1:epoch) {

    keras::unfreeze_weights(discriminator)

    for (j in 1:n.discriminator) {
      id <- sample(colnames(object),batch,replace = F)
      real <- object[,id]
      real1 <- t(real)

      id <- sample(colnames(object),batch,replace = F)
      real <- object[,id]
      real2 <- t(real)

      real_res <- array(runif(batch,-1, -1),dim =c(batch,1))
      d_loss_real <- discriminator%>%keras::train_on_batch(list(real1,real2),real_res)

      noise <- keras::k_random_normal(c(batch,l),mean = 0,stddev = 1)
      fake <- keras::predict_on_batch(generator,noise)
      fake_res <- array(runif(batch,1,1),dim =c(batch,1))
      d_loss_fake <-discriminator%>%keras::train_on_batch(list(real1,fake),fake_res)

      dloss[i] <- 0.5*( d_loss_fake+d_loss_real)

    }

    gc()
    keras::freeze_weights(discriminator)
    noise <- keras::k_random_normal(c(batch,l),mean = 0,stddev = 1)
    real_res <- array(runif(batch, -1 ,  -1),dim =c(batch,1))
    gloss[i]  <- GAN%>%keras::train_on_batch(list(noise,real1),real_res)


    fake <- keras::predict_on_batch(generator,noise)
    data_t <- t(rbind(fake,real1))
    x_corr <- Hmisc::rcorr(data_t,type = 'pearson')
    r <- x_corr$r
    r <- as.data.frame(r)[1:batch,batch+1:2*batch]
    g_coor_p[i] <- mean(as.matrix(r))


    print(i)
    print(  dloss[i])
    print(  gloss[i])
    print( g_coor_p[i] )
  }


}
