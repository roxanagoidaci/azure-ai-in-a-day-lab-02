using System;
using System.IO;
using Microsoft.Azure.WebJobs;
using Microsoft.Azure.WebJobs.Host;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Configuration;
using System.Net.Http;
using System.Net.Http.Headers;
using System.Text;
using System.Text.Json;

namespace COVID19Articles
{
    public static class COVID19ArticlesChangedTrigger
    {
        [FunctionName("COVID19ArticlesBlobUpdated")]
        public static async void Run([BlobTrigger("covid19articles/{name}", Connection = "dataset_STORAGE")]Stream myBlob, string name, ILogger log)
        {
            if (name == "COVID19Articles.csv")
            {
                log.LogInformation($"Dataset source blob storage updated\n Name:{name} \n Size: {myBlob.Length} Bytes");
                
                var webUri = System.Environment.GetEnvironmentVariable("AzureDevopsUri", EnvironmentVariableTarget.Process);
                
                var devopsPAT = System.Environment.GetEnvironmentVariable("AzureDevopsPAT", EnvironmentVariableTarget.Process);
                var buildDefinitionId = System.Environment.GetEnvironmentVariable("AzureDevopsBuildDefinitionId", EnvironmentVariableTarget.Process);
                var organization = System.Environment.GetEnvironmentVariable("AzureDevopsOrganization", EnvironmentVariableTarget.Process);
                var project = System.Environment.GetEnvironmentVariable("AzureDevopsProject", EnvironmentVariableTarget.Process);
                webUri = String.Format(webUri, buildDefinitionId, organization, project);
                
                using (var client = new HttpClient())
                {
                    client.BaseAddress = new Uri(webUri);
                    client.DefaultRequestHeaders.Accept.Add(new MediaTypeWithQualityHeaderValue("application/json"));
                    client.DefaultRequestHeaders.ConnectionClose = true;
                    client.DefaultRequestHeaders.Authorization = new AuthenticationHeaderValue("Basic", devopsPAT);

                    var request_json = "{\r\n  \"parameters\": {}, \"variables\": {} \r\n}";
    
                    var content = new StringContent(request_json, Encoding.UTF8, "application/json");

                    var httpresult = await client.PostAsJsonAsync(webUri, content);
                    string x = await httpresult.Content.ReadAsStringAsync();

                    log.LogInformation($"Devops response\n {x}");
                }
            }
        }
    }
}
