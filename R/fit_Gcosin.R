


#' Title Fits the model on data yielded batch-by-batch by Gcosin
#'
#' @param generator A generator (e.g. like the one provided by flow_images_from_directory() or a custom R generator function).
#' The output of the generator must be a list of one of these forms: <br> - (inputs, targets) <br> - (inputs, targets, sample_weights) <br>.
#'  This list (a single output of the generator) makes a single batch.
#'  Therefore, all arrays in this list must have the same length (equal to the size of this batch).
#'  Different batches may have different sizes. For example, the last batch of the epoch is commonly smaller than the others, if the size of the dataset is not divisible by the batch size.
#'  The generator is expected to loop over its data indefinitely. An epoch finishes when steps_per_epoch batches have been seen by the model.
#' @param batch Integer.Number of samples in each training
#' @param epoch Integer. Number of epochs to train the model.
#' An epoch is an iteration over the entire data provided,Note that in conjunction with initial_epoch,epochs is to be understood as “final epoch”.
#' The model is not trained for a number of iterations given by epochs,  but merely until the epoch of index epochs is reached.
#' @param learn_rate 	float >= 0. Learning rate.
#' @param object RNAseq matrix
#'
#' @return epoch, gloss, pearson correlations
#' @export
#'
#' @examples
#' \dontrun{
#' library(keras)
#' library(reticulate)
#' library(tensorflow)
#' library(bulkGAN)
#' library(Hmisc)
#'
#' fit_Gcosin(generator, batch=25,epoch = 500,
#'             learn_rate = 0.0001,object = normal_tissure )
#'}


fit_Gcosin <- function(generator, batch=25,epoch = 500,
                       learn_rate = 0.0001,object = NULL ){

  l <- 100
  gloss <- NULL
  g_coor_p <- NULL

  generator %>% keras::compile(optimizer = keras::optimizer_adamax(learning_rate = learn_rate),
                               loss = 'cosine_similarity' )

  for (i in 1:epoch) {
    noise <- keras::k_random_normal(c(batch,l),mean = 0,stddev = 1)
    id <- sample(colnames(object),batch,replace = T)
    real <- object[,id]
    real <- t(real)

    gloss[i]  <- generator%>%keras::train_on_batch( noise,real)


    fake <- keras::predict_on_batch(generator,noise)
    data_t <- t(rbind(fake,real))
    x_corr <- Hmisc::rcorr(data_t,type = 'pearson')
    r <- x_corr$r
    r <- as.data.frame(r)[1:batch,batch+1:2*batch]
    g_coor_p[i] <- mean(as.matrix(r))


    print(i)

    print(  gloss[i])
    print(g_coor_p[i])
  }


}

