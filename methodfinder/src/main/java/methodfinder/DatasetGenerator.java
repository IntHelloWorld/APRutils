package methodfinder;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;

import methodfinder.Common.CommandLineValues;
import methodfinder.Common.MethodContent;
import methodfinder.Common.ProgressBar;

public class DatasetGenerator {

    /**
     * Save extracted method files to output directory.
     * 
     * @throws InterruptedException
     */
    public static void generateDataset(ArrayList<MethodContent> methodContents, CommandLineValues commandLineValues,
            Path classPath, String projectName, String label, ProgressBar progressBar) throws InterruptedException {

        StringBuffer prefix = new StringBuffer();
        char splitWord = '#';
        prefix.append(label);
        prefix.append(splitWord);
        prefix.append(projectName);
        prefix.append(splitWord);
        prefix.append(classPath.toString().replace('/', '.'));
        prefix.append(splitWord);
        File dir = new File(commandLineValues.outputDir);
        if (!dir.exists()) {
            dir.mkdir();
        }

        for (MethodContent methodContent : methodContents) {
            StringBuffer fileName = new StringBuffer();
            fileName.append(prefix);
            fileName.append(methodContent.getName() + ".java");
            Path outPath = Paths.get(commandLineValues.outputDir, fileName.toString());
            try {
                File file = new File(outPath.toString());
                if (!file.exists()) {
                    file.createNewFile();
                }
                FileWriter fileWriter = new FileWriter(file);
                String methodCode = methodContent.getNode().toString();
                fileWriter.write(methodCode);
                fileWriter.close();
            } catch (IOException e) {
                e.printStackTrace();
                System.out.print("Output file write error!");
            }
        }
        progressBar.printProgress();
    }
}
