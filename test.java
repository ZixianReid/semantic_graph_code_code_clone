public class BreakInIfAndWhileExample {

    public static void main(String[] args) {
        int number = 1;

        // Using a while loop to find the first number divisible by both 3 and 5
        while (number <= 50) {
            if (number % 3 == 0 && number % 5 == 0) { // Condition inside if
                System.out.println("The first number divisible by 3 and 5 is: " + number);
                break; // Exit the while loop when condition is met
            }
            number++; // Increment the number
        }

        System.out.println("Loop exited.");
    }
}