pg <- read.csv("pointer-generator-analysis.csv", header=TRUE)
bu <- read.csv("bottom-up-analysis.csv", header=TRUE)
pg_correct_total <- sum(pg$correctly_marked_correct) +
  sum(pg$questionably_marked_correct) +
  sum(pg$incorrectly_marked_correct)
bu_correct_total <- sum(bu$correctly_marked_correct) +
  sum(bu$questionably_marked_correct) +
  sum(bu$incorrectly_marked_correct)
correct_total <- pg_correct_total + bu_correct_total
correctly_marked_correct <- sum(pg$correctly_marked_correct) +
  sum(bu$correctly_marked_correct)
questionably_marked_correct <- sum(pg$questionably_marked_correct) +
  sum(bu$questionably_marked_correct)
incorrectly_marked_correct <- sum(pg$incorrectly_marked_correct) +
  sum(bu$incorrectly_marked_correct)
pg_incorrect_total <- sum(pg$correctly_marked_incorrect) +
  sum(pg$questionably_marked_incorrect) +
  sum(pg$incorrectly_marked_incorrect)
bu_incorrect_total <- sum(bu$correctly_marked_incorrect) +
  sum(bu$questionably_marked_incorrect) +
  sum(bu$incorrectly_marked_incorrect)
incorrect_total <- pg_incorrect_total + bu_incorrect_total
correctly_marked_incorrect <- sum(pg$correctly_marked_incorrect) +
  sum(bu$correctly_marked_incorrect)
questionably_marked_incorrect <- sum(pg$questionably_marked_incorrect) +
  sum(bu$questionably_marked_incorrect)
incorrectly_marked_incorrect <- sum(pg$incorrectly_marked_incorrect) +
  sum(bu$incorrectly_marked_incorrect)
cat(sprintf("total marked correct: %d\n", correct_total))
cat(sprintf("%d %d %d\n", correctly_marked_correct,
            questionably_marked_correct, incorrectly_marked_correct))
cat(sprintf("total marked incorrect: %d\n", incorrect_total))
cat(sprintf("%d %d %d\n", correctly_marked_incorrect,
            questionably_marked_incorrect, incorrectly_marked_incorrect))
cat(sprintf("has explanation: %d %d %d\n",
            sum(pg$incorrectly_marked_incorrect_coref_issue) + 
            sum(bu$incorrectly_marked_incorrect_coref_issue),
            sum(pg$incorrectly_marked_incorrect_punct_issue) +
            sum(bu$incorrectly_marked_incorrect_punct_issue),
            sum(pg$incorrectly_marked_incorrect_conj_issue) +
            sum(bu$incorrectly_marked_incorrect_conj_issue)))
