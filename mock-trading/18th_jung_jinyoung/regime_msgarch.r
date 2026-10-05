# regime_msgarch.r — 2-레짐 MS-GJR-GARCH 추정 + 필터링 확률 (reference/02_regime.r 기반)
#
# 사용: Rscript regime_msgarch.r <input.csv> <train_end YYYY-MM-DD> <out_prob.csv> <fit.rds> <n_seeds> [filter_only]
#   input.csv : date, ret (퍼센트 단위 시장 수익률)
#   train_end 까지로 추정(또는 filter_only면 fit.rds 재사용), 전체 구간 필터링 확률 출력
#   스무딩 확률은 쓰지 않음.

suppressMessages(library(MSGARCH))
a <- commandArgs(trailingOnly = TRUE)
inp <- a[1]; train_end <- as.Date(a[2]); outp <- a[3]; rds <- a[4]; nseed <- as.integer(a[5])
filter_only <- length(a) >= 6 && a[6] == "filter_only"

d <- read.csv(inp); d$date <- as.Date(d$date)
r_all <- d$ret
is_tr <- d$date <= train_end
r_train <- r_all[is_tr]; r_new <- r_all[!is_tr]

spec <- CreateSpec(variance.spec = list(model = "gjrGARCH"),
                   distribution.spec = list(distribution = "std"),
                   switch.spec = list(K = 2))

if (filter_only) {
  saved <- readRDS(rds)                       # 추정 파라미터만 저장 (Rcpp 객체는 세션 간 재사용 불가)
  if (saved$n_train != length(r_train)) stop("filter_only: 학습 데이터 길이가 저장된 추정과 다름")
  par <- saved$par; ll <- saved$ll
} else {
  fits <- lapply(seq_len(nseed), function(s) { set.seed(s); tryCatch(FitML(spec, data = r_train), error = function(e) NULL) })
  ok <- !sapply(fits, is.null)
  ll <- sapply(fits[ok], function(f) f$loglik)
  par <- fits[ok][[which.max(ll)]]$par
  saveRDS(list(par = par, n_train = length(r_train), ll = ll), rds)
}

# 고정 파라미터로 전 구간 필터링 (필터링 확률은 t까지의 데이터만 사용)
st <- State(object = spec, par = par, data = r_all)
fp <- st$FiltProb[, 1, ]
if (nrow(fp) != length(r_all)) stop(sprintf("State() 길이 %d != 입력 길이 %d", nrow(fp), length(r_all)))

rv <- sapply(1:2, function(k) sum(fp[is_tr, k] * r_train^2) / sum(fp[is_tr, k]))
turb <- which.max(rv); calm <- 3 - turb

out <- data.frame(date = d$date, p_turb = fp[, turb])
write.csv(out, outp, row.names = FALSE)
cat(sprintf("LL_MAX=%.3f LL_RANGE=%.3f NSEED_OK=%d VOL_CALM=%.2f VOL_TURB=%.2f\n",
            max(ll), diff(range(ll)), length(ll), sqrt(rv[calm] * 252), sqrt(rv[turb] * 252)))
