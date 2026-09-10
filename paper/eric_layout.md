title: something cool as a palindrome, "Revolt, Academia: Aimed a Cat Lover"
comma, finding longer palindromes with meaning

A palindrome is a word or phrase which reads the same backwards as forwards, not counting any capitalization nor punctuation. For example; "racecar", "taco cat", and "Non-academia aimed a canon" are all palindromes.  Palindromes are easy to verify. Take the first letter and the last letter of the phrase and count one forward and one backwards respectively, making sure that each letter is the same, until you reach the other side. [photo of counting a palindrome, maybe]

Given verifying a palindrome is both fast and programmatic, the search of readable palindromes therefore needs to be a reduction of the broad search into something humans can read.
For example look at this palindrome which is no doubt a palindrome, but it is unreadable given that no one can derive any meaning from it.
The goal is to find long palindrome phrases that read as real English prose.

the mechanical condition is our verifier and length, the search function
the non-mechanical conditions are our readability, coherence, and meaning
while "racecar racecar racecar racecar" is technically a long palindrome, it has no interesting meaning in our case

so, there are a couple ways to find palindromes
Hoey and Norvig in 2002 had developed the idea of overhangs in palindrome search
[explain how one word has debt the other side needs to recover]
[taking too much debt can be okay, as long as it is held up, know the limits of debt]
We define a seam as a couple of known palindromes coupled together through 
